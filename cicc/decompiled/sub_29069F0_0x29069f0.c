// Function: sub_29069F0
// Address: 0x29069f0
//
__int64 __fastcall sub_29069F0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r9
  __int64 v13; // r13
  __int64 v14; // rbx
  _QWORD *v15; // rax
  _QWORD *v16; // r14
  int v18; // edx
  int v19; // r10d
  __int64 v20; // [rsp+8h] [rbp-58h] BYREF
  char *v21; // [rsp+10h] [rbp-50h] BYREF
  char v22; // [rsp+30h] [rbp-30h]
  char v23; // [rsp+31h] [rbp-2Fh]

  v5 = sub_2906530(a2, *a1, a1[1]);
  v6 = a1[2];
  v20 = v5;
  v7 = v5;
  v8 = *(unsigned int *)(v6 + 24);
  v9 = *(_QWORD *)(v6 + 8);
  if ( (_DWORD)v8 )
  {
    v10 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( v7 == *v11 )
    {
LABEL_3:
      if ( v11 != (__int64 *)(v9 + 16 * v8) )
        v7 = *(_QWORD *)(sub_2904570(v6, &v20) + 48);
    }
    else
    {
      v18 = 1;
      while ( v12 != -4096 )
      {
        v19 = v18 + 1;
        v10 = (v8 - 1) & (v18 + v10);
        v11 = (__int64 *)(v9 + 16LL * v10);
        v12 = *v11;
        if ( v7 == *v11 )
          goto LABEL_3;
        v18 = v19;
      }
    }
  }
  v13 = *(_QWORD *)(a2 + 8);
  if ( *(_QWORD *)(v7 + 8) != v13 && a3 )
  {
    v23 = 1;
    v14 = a3 + 24;
    v21 = "cast";
    v22 = 3;
    v15 = sub_BD2C40(72, 1u);
    v16 = v15;
    if ( v15 )
      sub_B51BF0((__int64)v15, v7, v13, (__int64)&v21, v14, 0);
    return (__int64)v16;
  }
  return v7;
}
