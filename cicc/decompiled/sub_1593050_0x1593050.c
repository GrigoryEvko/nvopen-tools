// Function: sub_1593050
// Address: 0x1593050
//
__int64 __fastcall sub_1593050(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rdi
  bool v11; // cc
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned int v15; // edx
  __int64 v17; // [rsp+0h] [rbp-B0h]
  __int64 v18; // [rsp+8h] [rbp-A8h]
  __int64 v19; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-88h]
  __int64 v21; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v22; // [rsp+38h] [rbp-78h]
  __int64 v23; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+48h] [rbp-68h]
  __int64 v25; // [rsp+50h] [rbp-60h]
  unsigned int v26; // [rsp+58h] [rbp-58h]
  __int64 v27; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+68h] [rbp-48h]
  __int64 v29; // [rsp+70h] [rbp-40h]
  int v30; // [rsp+78h] [rbp-38h]

  v4 = *(unsigned int *)(a2 + 8);
  v5 = *(_DWORD *)(a2 + 8) >> 1;
  v6 = *(_QWORD *)(*(_QWORD *)(a2 - 8 * v4) + 136LL);
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 8 * (1 - v4)) + 136LL);
  v28 = *(_DWORD *)(v7 + 32);
  if ( v28 > 0x40 )
    sub_16A4FD0(&v27, v7 + 24);
  else
    v27 = *(_QWORD *)(v7 + 24);
  v24 = *(_DWORD *)(v6 + 32);
  if ( v24 > 0x40 )
    sub_16A4FD0(&v23, v6 + 24);
  else
    v23 = *(_QWORD *)(v6 + 24);
  sub_15898E0(a1, (__int64)&v23, &v27);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( (unsigned int)v5 > 1 )
  {
    v8 = 2 * v5;
    v9 = 2;
    v18 = v8;
    do
    {
      v12 = *(unsigned int *)(a2 + 8);
      v13 = *(_QWORD *)(*(_QWORD *)(a2 + 8 * (v9 - v12)) + 136LL);
      v14 = *(_QWORD *)(*(_QWORD *)(a2 + 8 * (v9 + 1 - v12)) + 136LL);
      v22 = *(_DWORD *)(v14 + 32);
      if ( v22 > 0x40 )
      {
        v17 = v13;
        sub_16A4FD0(&v21, v14 + 24);
        v13 = v17;
      }
      else
      {
        v21 = *(_QWORD *)(v14 + 24);
      }
      v20 = *(_DWORD *)(v13 + 32);
      if ( v20 > 0x40 )
        sub_16A4FD0(&v19, v13 + 24);
      else
        v19 = *(_QWORD *)(v13 + 24);
      sub_15898E0((__int64)&v23, (__int64)&v19, &v21);
      sub_158C3A0((__int64)&v27, a1, (__int64)&v23);
      if ( *(_DWORD *)(a1 + 8) > 0x40u && *(_QWORD *)a1 )
        j_j___libc_free_0_0(*(_QWORD *)a1);
      v11 = *(_DWORD *)(a1 + 24) <= 0x40u;
      *(_QWORD *)a1 = v27;
      v15 = v28;
      v28 = 0;
      *(_DWORD *)(a1 + 8) = v15;
      if ( v11 || (v10 = *(_QWORD *)(a1 + 16)) == 0 )
      {
        *(_QWORD *)(a1 + 16) = v29;
        *(_DWORD *)(a1 + 24) = v30;
      }
      else
      {
        j_j___libc_free_0_0(v10);
        v11 = v28 <= 0x40;
        *(_QWORD *)(a1 + 16) = v29;
        *(_DWORD *)(a1 + 24) = v30;
        if ( !v11 && v27 )
          j_j___libc_free_0_0(v27);
      }
      if ( v26 > 0x40 && v25 )
        j_j___libc_free_0_0(v25);
      if ( v24 > 0x40 && v23 )
        j_j___libc_free_0_0(v23);
      if ( v20 > 0x40 && v19 )
        j_j___libc_free_0_0(v19);
      if ( v22 > 0x40 && v21 )
        j_j___libc_free_0_0(v21);
      v9 += 2;
    }
    while ( v18 != v9 );
  }
  return a1;
}
