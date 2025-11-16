// Function: sub_9AC9C0
// Address: 0x9ac9c0
//
__int64 __fastcall sub_9AC9C0(__int64 a1, unsigned __int64 a2, __m128i *a3)
{
  unsigned __int64 v4; // rax
  unsigned __int16 v5; // ax
  unsigned int v6; // r12d
  __int64 v8; // rcx
  int v9; // eax
  __int64 v10; // [rsp+0h] [rbp-A0h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-98h]
  __int64 v12; // [rsp+10h] [rbp-90h]
  unsigned int v13; // [rsp+18h] [rbp-88h]
  __int64 v14; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-78h]
  __int64 v16; // [rsp+30h] [rbp-70h]
  unsigned int v17; // [rsp+38h] [rbp-68h]
  unsigned __int64 v18; // [rsp+40h] [rbp-60h] BYREF
  __int64 v19; // [rsp+48h] [rbp-58h]
  unsigned int v20; // [rsp+50h] [rbp-50h]
  __int64 v21; // [rsp+58h] [rbp-48h]
  unsigned int v22; // [rsp+60h] [rbp-40h]

  v4 = *(unsigned __int8 *)a2;
  if ( (_BYTE)v4 != 51 )
  {
    if ( (unsigned __int8)v4 > 0x1Cu )
    {
      if ( (unsigned __int8)v4 > 0x36u )
        goto LABEL_4;
      v8 = 0x40540000000000LL;
      if ( !_bittest64(&v8, v4) )
        goto LABEL_4;
      v9 = (unsigned __int8)v4 - 29;
    }
    else
    {
      if ( (_BYTE)v4 != 5 )
        goto LABEL_4;
      v9 = *(unsigned __int16 *)(a2 + 2);
      if ( (*(_WORD *)(a2 + 2) & 0xFFF7) != 0x11 && (v9 & 0xFFFD) != 0xD )
        goto LABEL_4;
    }
    if ( v9 != 15 || (*(_BYTE *)(a2 + 1) & 2) == 0 )
      goto LABEL_4;
  }
  if ( a1 == *(_QWORD *)(a2 - 64)
    && sub_98EF80((unsigned __int8 *)a1, a3[2].m128i_i64[0], a3[2].m128i_i64[1], a3[1].m128i_i64[1], 0) )
  {
    return 3;
  }
LABEL_4:
  v5 = sub_9A1D50(0x23u, a1, (unsigned __int8 *)a2, a3[2].m128i_i64[1], a3->m128i_i64[0]);
  if ( HIBYTE(v5) )
  {
    return (_BYTE)v5 != 0 ? 3 : 0;
  }
  else
  {
    v18 = a1 & 0xFFFFFFFFFFFFFFFBLL;
    v20 = 1;
    v19 = 0;
    v22 = 1;
    v21 = 0;
    sub_9AC780((__int64)&v10, (__int64 *)&v18, 0, a3);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    if ( v20 > 0x40 && v19 )
      j_j___libc_free_0_0(v19);
    v18 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    v20 = 1;
    v19 = 0;
    v22 = 1;
    v21 = 0;
    sub_9AC780((__int64)&v14, (__int64 *)&v18, 0, a3);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    if ( v20 > 0x40 && v19 )
      j_j___libc_free_0_0(v19);
    v6 = sub_ABE130(&v10, &v14);
    if ( v6 > 3 )
      BUG();
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
    if ( v11 > 0x40 && v10 )
      j_j___libc_free_0_0(v10);
  }
  return v6;
}
