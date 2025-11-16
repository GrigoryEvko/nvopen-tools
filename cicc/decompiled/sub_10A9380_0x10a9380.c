// Function: sub_10A9380
// Address: 0x10a9380
//
__int64 __fastcall sub_10A9380(_QWORD **a1, int a2, unsigned __int8 *a3, _QWORD *a4, __int64 a5)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // rcx
  __int64 v12; // r8
  unsigned __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  char v16; // al
  unsigned __int64 v17; // rsi
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp-28h] [rbp-28h]
  __int64 v22; // [rsp-28h] [rbp-28h]

  if ( a2 + 29 != *a3 )
    return 0;
  v6 = *((_QWORD *)a3 - 8);
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 || *(_QWORD *)(v7 + 8) || *(_BYTE *)v6 != 47 )
    goto LABEL_4;
  v22 = (__int64)a3;
  v16 = sub_995E90(a1, *(_QWORD *)(v6 - 64), (__int64)a3, (__int64)a4, a5);
  v17 = *(_QWORD *)(v6 - 32);
  a3 = (unsigned __int8 *)v22;
  if ( v16 && v17 )
  {
    *a1[1] = v17;
    goto LABEL_19;
  }
  v18 = sub_995E90(a1, v17, v22, (__int64)a4, a5);
  a3 = (unsigned __int8 *)v22;
  if ( !v18 || (v19 = *(_QWORD *)(v6 - 64)) == 0 )
  {
LABEL_4:
    v8 = *((_QWORD *)a3 - 4);
LABEL_5:
    v9 = *(_QWORD *)(v8 + 16);
    if ( !v9 || *(_QWORD *)(v9 + 8) || *(_BYTE *)v8 != 47 )
      return 0;
    v21 = (__int64)a3;
    v10 = sub_995E90(a1, *(_QWORD *)(v8 - 64), (__int64)a3, (__int64)a4, a5);
    v13 = *(_QWORD *)(v8 - 32);
    v14 = v21;
    if ( v10 && v13 )
    {
      *a1[1] = v13;
    }
    else
    {
      if ( !(unsigned __int8)sub_995E90(a1, v13, v21, v11, v12) )
        return 0;
      v20 = *(_QWORD *)(v8 - 64);
      if ( !v20 )
        return 0;
      v14 = v21;
      *a1[1] = v20;
    }
    v15 = *(_QWORD *)(v14 - 64);
    if ( v15 )
    {
      *a1[2] = v15;
      return 1;
    }
    return 0;
  }
  a4 = a1[1];
  *a4 = v19;
LABEL_19:
  v8 = *((_QWORD *)a3 - 4);
  if ( !v8 )
    goto LABEL_5;
  *a1[2] = v8;
  return 1;
}
