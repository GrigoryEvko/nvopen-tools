// Function: sub_9AFB10
// Address: 0x9afb10
//
__int64 __fastcall sub_9AFB10(__int64 a1, __int64 a2, __m128i *a3)
{
  unsigned __int64 v4; // rax
  unsigned int v5; // r12d
  __int64 v7; // rcx
  int v8; // eax
  __int64 v9; // [rsp+0h] [rbp-A0h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-98h]
  __int64 v11; // [rsp+10h] [rbp-90h]
  unsigned int v12; // [rsp+18h] [rbp-88h]
  __int64 v13; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-78h]
  __int64 v15; // [rsp+30h] [rbp-70h]
  unsigned int v16; // [rsp+38h] [rbp-68h]
  unsigned __int64 v17; // [rsp+40h] [rbp-60h] BYREF
  __int64 v18; // [rsp+48h] [rbp-58h]
  unsigned int v19; // [rsp+50h] [rbp-50h]
  __int64 v20; // [rsp+58h] [rbp-48h]
  unsigned int v21; // [rsp+60h] [rbp-40h]

  v4 = *(unsigned __int8 *)a2;
  if ( (_BYTE)v4 != 52 )
  {
    if ( (unsigned __int8)v4 > 0x1Cu )
    {
      if ( (unsigned __int8)v4 > 0x36u )
        goto LABEL_4;
      v7 = 0x40540000000000LL;
      if ( !_bittest64(&v7, v4) )
        goto LABEL_4;
      v8 = (unsigned __int8)v4 - 29;
    }
    else
    {
      if ( (_BYTE)v4 != 5 )
        goto LABEL_4;
      v8 = *(unsigned __int16 *)(a2 + 2);
      if ( (*(_WORD *)(a2 + 2) & 0xFFF7) != 0x11 && (v8 & 0xFFFD) != 0xD )
        goto LABEL_4;
    }
    if ( v8 != 15 || (*(_BYTE *)(a2 + 1) & 4) == 0 )
      goto LABEL_4;
  }
  if ( a1 == *(_QWORD *)(a2 - 64)
    && sub_98EF80((unsigned __int8 *)a1, a3[2].m128i_i64[0], a3[2].m128i_i64[1], a3[1].m128i_i64[1], 0) )
  {
    return 3;
  }
LABEL_4:
  if ( (unsigned int)sub_9AF7E0(a1, 0, a3) > 1 && (unsigned int)sub_9AF7E0(a2, 0, a3) > 1 )
    return 3;
  v17 = a1 & 0xFFFFFFFFFFFFFFFBLL;
  v19 = 1;
  v18 = 0;
  v21 = 1;
  v20 = 0;
  sub_9AC780((__int64)&v9, (__int64 *)&v17, 1u, a3);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  v17 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v19 = 1;
  v18 = 0;
  v21 = 1;
  v20 = 0;
  sub_9AC780((__int64)&v13, (__int64 *)&v17, 1u, a3);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  v5 = sub_ABE240(&v9, &v13);
  if ( v5 > 3 )
    BUG();
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  return v5;
}
