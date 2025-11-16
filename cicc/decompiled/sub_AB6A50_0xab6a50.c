// Function: sub_AB6A50
// Address: 0xab6a50
//
__int64 __fastcall sub_AB6A50(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r15d
  bool v6; // r15
  unsigned int v7; // edx
  int v8; // eax
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned int v11; // r13d
  __int64 v12; // r12
  int v13; // eax
  _QWORD *v14; // rdx
  unsigned int v15; // [rsp+0h] [rbp-80h]
  unsigned int v16; // [rsp+4h] [rbp-7Ch]
  unsigned int v17; // [rsp+4h] [rbp-7Ch]
  __int64 v18; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-68h]
  _QWORD *v20; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-58h]
  __int64 v22; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-48h]
  __int64 v24; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v25; // [rsp+48h] [rbp-38h]

  if ( sub_AAF7D0(a2) || sub_AAF7D0(a3) )
    goto LABEL_2;
  sub_AB0910((__int64)&v24, a3);
  v5 = v25;
  if ( v25 <= 0x40 )
  {
    v6 = v24 == 0;
  }
  else
  {
    v6 = v5 == (unsigned int)sub_C444A0(&v24);
    if ( v24 )
      j_j___libc_free_0_0(v24);
  }
  if ( v6 )
  {
LABEL_2:
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  sub_AB0A00((__int64)&v22, a2);
  sub_AB0910((__int64)&v24, a3);
  sub_C4A1D0(&v18, &v22, &v24);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  sub_AB0A00((__int64)&v20, a3);
  v7 = v21;
  if ( v21 <= 0x40 )
  {
    if ( v20 )
      goto LABEL_24;
  }
  else
  {
    v16 = v21;
    v8 = sub_C444A0(&v20);
    v7 = v16;
    if ( v16 != v8 )
      goto LABEL_24;
  }
  v17 = *(_DWORD *)(a3 + 24);
  if ( v17 > 0x40 )
  {
    v15 = v7;
    v13 = sub_9871A0(a3 + 16);
    v7 = v15;
    if ( v17 - v13 > 0x40 )
    {
LABEL_20:
      if ( v7 > 0x40 )
      {
        *v20 = 1;
        memset(v20 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v21 + 63) >> 6) - 8);
      }
      else
      {
        v10 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & 1;
        if ( !v7 )
          v10 = 0;
        v20 = (_QWORD *)v10;
      }
      goto LABEL_24;
    }
    v9 = **(_QWORD **)(a3 + 16);
  }
  else
  {
    v9 = *(_QWORD *)(a3 + 16);
  }
  if ( v9 != 1 )
    goto LABEL_20;
  if ( v7 <= 0x40 && *(_DWORD *)(a3 + 8) <= 0x40u )
  {
    v14 = *(_QWORD **)a3;
    v21 = *(_DWORD *)(a3 + 8);
    v20 = v14;
  }
  else
  {
    sub_C43990(&v20, a3);
  }
LABEL_24:
  sub_AB0910((__int64)&v22, a2);
  sub_C4A1D0(&v24, &v22, &v20);
  sub_C46A40(&v24, 1);
  v11 = v25;
  v12 = v24;
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  v25 = v11;
  v23 = v19;
  v24 = v12;
  v22 = v18;
  v19 = 0;
  sub_9875E0(a1, &v22, &v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return a1;
}
