// Function: sub_15A11D0
// Address: 0x15a11d0
//
__int64 __fastcall sub_15A11D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, float a5)
{
  unsigned int v5; // r14d
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v17; // r8
  __int64 v18; // r12
  __int64 v19; // rsi
  __int64 v20; // rbx
  __int64 v21; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+8h] [rbp-58h]
  _BYTE v23[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v24; // [rsp+18h] [rbp-48h] BYREF
  __int64 v25; // [rsp+20h] [rbp-40h]

  v5 = a3;
  v7 = *(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)v7 == 16 )
    v7 = *(unsigned __int8 *)(**(_QWORD **)(a1 + 16) + 8LL);
  v8 = sub_1593350(v7, a2, a3, a4);
  v11 = sub_16982C0(v7, a2, v9, v10);
  v12 = v11;
  if ( v5 )
  {
    v22 = 64;
    v21 = v5;
    if ( v8 == v11 )
      sub_169C580(&v24, v11, 0);
    else
      sub_1698390(&v24, v8, 0);
    if ( v12 == v24 )
      sub_169CAA0(&v24, 0, (unsigned __int8)a2, &v21, v13, a5);
    else
      sub_16986F0(&v24, 0, (unsigned __int8)a2, &v21, v13, a5);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
  }
  else
  {
    if ( v8 == v11 )
      sub_169C580(&v24, v11, 0);
    else
      sub_1698390(&v24, v8, 0);
    if ( v12 == v24 )
      sub_169CAA0(&v24, 0, (unsigned __int8)a2, 0, v17, a5);
    else
      sub_16986F0(&v24, 0, (unsigned __int8)a2, 0, v17, a5);
  }
  v14 = sub_159CCF0(*(_QWORD **)a1, (__int64)v23);
  v15 = v14;
  if ( *(_BYTE *)(a1 + 8) == 16 )
    v15 = sub_15A0390(*(_QWORD *)(a1 + 32), v14);
  if ( v12 == v24 )
  {
    v18 = v25;
    if ( v25 )
    {
      v19 = 32LL * *(_QWORD *)(v25 - 8);
      v20 = v25 + v19;
      if ( v25 != v25 + v19 )
      {
        do
        {
          v20 -= 32;
          sub_127D120((_QWORD *)(v20 + 8));
        }
        while ( v18 != v20 );
      }
      j_j_j___libc_free_0_0(v18 - 8);
    }
  }
  else
  {
    sub_1698460(&v24);
  }
  return v15;
}
