// Function: sub_3715140
// Address: 0x3715140
//
__int64 *__fastcall sub_3715140(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r12
  __int64 *v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int64 v14; // rax
  __int64 v15; // [rsp+18h] [rbp-A8h] BYREF
  __m128i v16; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v17; // [rsp+30h] [rbp-90h] BYREF
  __int64 v18[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v19; // [rsp+50h] [rbp-70h] BYREF
  __m128i v20[2]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v21; // [rsp+80h] [rbp-40h]

  v5 = (_QWORD *)(a2 + 16);
  v6 = sub_3707A00();
  sub_37145A0(&v16, (_QWORD *)(a2 + 16), *(_WORD *)(a4 + 4), (unsigned __int64)v6, v7, (__int64)&v16);
  v20[0].m128i_i64[0] = (__int64)"MemberCount";
  v21 = 259;
  sub_370BC10((unsigned __int64 *)v18, (_QWORD *)(a2 + 16), (unsigned __int16 *)(a4 + 2), v20);
  if ( (v18[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v18[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    sub_8FD6D0((__int64)v18, "Properties", &v16);
    v21 = 260;
    v20[0].m128i_i64[0] = (__int64)v18;
    sub_370F7D0((unsigned __int64 *)&v15, v5, (unsigned __int16 *)(a4 + 4), v20, (unsigned int)&v15);
    if ( (__int64 *)v18[0] != &v19 )
      j_j___libc_free_0(v18[0]);
    v8 = v15 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v15 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v15 = 0;
      *a1 = v8 | 1;
      sub_9C66B0(&v15);
    }
    else
    {
      v15 = 0;
      sub_9C66B0(&v15);
      v20[0].m128i_i64[0] = (__int64)"FieldList";
      v21 = 259;
      sub_37011E0((unsigned __int64 *)v18, v5, (unsigned int *)(a4 + 6), v20[0].m128i_i64);
      v10 = v18[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v18[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v18[0] = 0;
        *a1 = v10 | 1;
        sub_9C66B0(v18);
      }
      else
      {
        v18[0] = 0;
        sub_9C66B0(v18);
        v20[0].m128i_i64[0] = (__int64)"DerivedFrom";
        v21 = 259;
        sub_37011E0((unsigned __int64 *)v18, v5, (unsigned int *)(a4 + 48), v20[0].m128i_i64);
        v11 = v18[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_18;
        v18[0] = 0;
        sub_9C66B0(v18);
        v20[0].m128i_i64[0] = (__int64)"VShape";
        v21 = 259;
        sub_37011E0((unsigned __int64 *)v18, v5, (unsigned int *)(a4 + 52), v20[0].m128i_i64);
        v11 = v18[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (v18[0] & 0xFFFFFFFFFFFFFFFELL) != 0
          || (v18[0] = 0,
              sub_9C66B0(v18),
              v20[0].m128i_i64[0] = (__int64)"SizeOf",
              v21 = 259,
              sub_3702660((unsigned __int64 *)v18, v5, (unsigned __int64 *)(a4 + 56), v20, v12, v13),
              v11 = v18[0] & 0xFFFFFFFFFFFFFFFELL,
              (v18[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
        {
LABEL_18:
          *a1 = 0;
          v18[0] = v11 | 1;
          sub_9C6670(a1, v18);
          sub_9C66B0(v18);
        }
        else
        {
          v18[0] = 0;
          sub_9C66B0(v18);
          sub_370D750(v20[0].m128i_i64, v5, (int **)(a4 + 16), (int **)(a4 + 32), (*(_WORD *)(a4 + 4) & 0x200) != 0);
          v14 = v20[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (v20[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            *a1 = 0;
            v20[0].m128i_i64[0] = v14 | 1;
            sub_9C6670(a1, v20);
          }
          else
          {
            v20[0].m128i_i64[0] = 0;
            sub_9C66B0(v20[0].m128i_i64);
            *a1 = 1;
          }
          sub_9C66B0(v20[0].m128i_i64);
        }
      }
    }
  }
  if ( (__int64 *)v16.m128i_i64[0] != &v17 )
    j_j___libc_free_0(v16.m128i_u64[0]);
  return a1;
}
