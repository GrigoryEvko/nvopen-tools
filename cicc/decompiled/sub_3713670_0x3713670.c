// Function: sub_3713670
// Address: 0x3713670
//
__int64 *__fastcall sub_3713670(__int64 *a1, _BYTE *a2, _QWORD *a3, __int64 a4)
{
  unsigned __int64 v6; // rax
  unsigned int v7; // r8d
  unsigned __int64 v8; // rax
  unsigned __int64 v10; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v11[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v12; // [rsp+30h] [rbp-90h] BYREF
  __int64 v13[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v14; // [rsp+50h] [rbp-70h] BYREF
  __m128i v15[2]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v16; // [rsp+80h] [rbp-40h]

  sub_37128E0(v11, a3, *(_WORD *)(a4 + 6) & 3, (*(_WORD *)(a4 + 6) >> 2) & 7, *(_WORD *)(a4 + 6) & 0xFFE0);
  sub_8FD6D0((__int64)v13, "Attrs: ", v11);
  v15[0].m128i_i64[0] = (__int64)v13;
  v16 = 260;
  sub_370BC10(&v10, a3, (unsigned __int16 *)(a4 + 6), v15);
  if ( (__int64 *)v13[0] != &v14 )
    j_j___libc_free_0(v13[0]);
  if ( (v10 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v10 & 0xFFFFFFFFFFFFFFFELL | 1;
    goto LABEL_16;
  }
  if ( *a2 )
  {
    v16 = 257;
    LOWORD(v10) = 0;
    sub_370BC10((unsigned __int64 *)v13, a3, (unsigned __int16 *)&v10, v15);
    v6 = v13[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v13[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
LABEL_19:
      v13[0] = 0;
      *a1 = v6 | 1;
      sub_9C66B0(v13);
      goto LABEL_16;
    }
    v13[0] = 0;
    sub_9C66B0(v13);
  }
  v15[0].m128i_i64[0] = (__int64)"Type";
  v16 = 259;
  sub_37011E0((unsigned __int64 *)v13, a3, (unsigned int *)(a4 + 2), v15[0].m128i_i64);
  v6 = v13[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v13[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_19;
  v13[0] = 0;
  sub_9C66B0(v13);
  if ( ((*(_WORD *)(a4 + 6) >> 2) & 5) != 4 )
  {
    if ( a3[5] && !a3[7] && !a3[6] )
      *(_DWORD *)(a4 + 8) = -1;
    goto LABEL_11;
  }
  v15[0].m128i_i64[0] = (__int64)"VFTableOffset";
  v16 = 259;
  sub_370BFD0((unsigned __int64 *)v13, a3, (unsigned int *)(a4 + 8), v15);
  v8 = v13[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v13[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    v13[0] = 0;
    sub_9C66B0(v13);
LABEL_11:
    if ( *a2 )
    {
LABEL_14:
      *a1 = 1;
      v15[0].m128i_i64[0] = 0;
      sub_9C66B0(v15[0].m128i_i64);
      goto LABEL_16;
    }
    v15[0].m128i_i64[0] = (__int64)"Name";
    v16 = 259;
    sub_3701560((unsigned __int64 *)v13, a3, (_QWORD *)(a4 + 16), v15, v7);
    v8 = v13[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v13[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      v13[0] = 0;
      sub_9C66B0(v13);
      goto LABEL_14;
    }
  }
  *a1 = 0;
  v13[0] = v8 | 1;
  sub_9C6670(a1, v13);
  sub_9C66B0(v13);
LABEL_16:
  if ( (__int64 *)v11[0] != &v12 )
    j_j___libc_free_0(v11[0]);
  return a1;
}
