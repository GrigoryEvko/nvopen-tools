// Function: sub_2B05B90
// Address: 0x2b05b90
//
char __fastcall sub_2B05B90(__int64 a1, __int64 a2, char *a3, char *a4)
{
  char *v5; // rsi
  char *v6; // rdx
  __int64 v7; // r8
  unsigned int v8; // eax
  unsigned int v9; // ecx
  bool v10; // cf
  __int64 v12; // r10
  unsigned int v13; // r9d
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdi
  int v17; // ebx
  __int64 v18; // r9
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdx
  _QWORD *v21; // rcx
  _QWORD *v22; // rax
  __int64 v23; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v24[5]; // [rsp+18h] [rbp-28h] BYREF

  v23 = a2;
  v24[0] = &v23;
  v5 = sub_2AF7450(a3);
  v6 = sub_2AF7450(a4);
  if ( !((unsigned __int64)v6 | (unsigned __int64)v5) )
  {
LABEL_10:
    v17 = sub_2B05830(v24, v7);
    return v17 < (int)sub_2B05830(v24, (__int64)a4);
  }
  if ( !v5 )
    return 1;
  if ( !v6 )
    return 0;
  v8 = *((_DWORD *)v5 + 1) & 0x7FFFFFF;
  v9 = *((_DWORD *)v6 + 1) & 0x7FFFFFF;
  v10 = v8 < v9;
  if ( v8 != v9 )
    return v10;
  v12 = v8;
  v13 = v8 - 1;
  if ( v8 != 1 )
  {
    v14 = v8 - 2;
    v15 = 0;
    v16 = 32 * (v14 + 1);
    while ( *(_QWORD *)&v6[v15 + -32 * v12] == *(_QWORD *)&v5[v15 + -32 * v12] )
    {
      v15 += 32;
      if ( v16 == v15 )
        goto LABEL_13;
    }
    goto LABEL_10;
  }
LABEL_13:
  v18 = 32 * (v13 - v12);
  v19 = *(_QWORD *)&v5[v18];
  v20 = *(_QWORD *)&v6[v18];
  if ( *(_BYTE *)v19 != 17 || *(_BYTE *)v20 != 17 )
    return v19 < v20;
  v21 = *(_QWORD **)(v19 + 24);
  if ( *(_DWORD *)(v19 + 32) > 0x40u )
    v21 = (_QWORD *)*v21;
  v22 = *(_QWORD **)(v20 + 24);
  if ( *(_DWORD *)(v20 + 32) > 0x40u )
    v22 = (_QWORD *)*v22;
  return v22 > v21;
}
