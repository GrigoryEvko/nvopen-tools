// Function: sub_3715750
// Address: 0x3715750
//
__int64 *__fastcall sub_3715750(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r12
  __int64 *v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // [rsp+18h] [rbp-A8h] BYREF
  __m128i v14; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v15; // [rsp+30h] [rbp-90h] BYREF
  __int64 v16[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v17; // [rsp+50h] [rbp-70h] BYREF
  __m128i v18[2]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v19; // [rsp+80h] [rbp-40h]

  v5 = (_QWORD *)(a2 + 16);
  v6 = sub_3707A00();
  sub_37145A0(&v14, (_QWORD *)(a2 + 16), *(_WORD *)(a4 + 4), (unsigned __int64)v6, v7, (__int64)&v14);
  v18[0].m128i_i64[0] = (__int64)"NumEnumerators";
  v19 = 259;
  sub_370BC10((unsigned __int64 *)v16, (_QWORD *)(a2 + 16), (unsigned __int16 *)(a4 + 2), v18);
  if ( (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v16[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    sub_8FD6D0((__int64)v16, "Properties", &v14);
    v19 = 260;
    v18[0].m128i_i64[0] = (__int64)v16;
    sub_370F7D0((unsigned __int64 *)&v13, v5, (unsigned __int16 *)(a4 + 4), v18, (unsigned int)&v13);
    if ( (__int64 *)v16[0] != &v17 )
      j_j___libc_free_0(v16[0]);
    v8 = v13 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v13 = 0;
      *a1 = v8 | 1;
      sub_9C66B0(&v13);
    }
    else
    {
      v13 = 0;
      sub_9C66B0(&v13);
      v18[0].m128i_i64[0] = (__int64)"UnderlyingType";
      v19 = 259;
      sub_37011E0((unsigned __int64 *)v16, v5, (unsigned int *)(a4 + 48), v18[0].m128i_i64);
      v10 = v16[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v16[0] = 0;
        *a1 = v10 | 1;
        sub_9C66B0(v16);
      }
      else
      {
        v16[0] = 0;
        sub_9C66B0(v16);
        v18[0].m128i_i64[0] = (__int64)"FieldListType";
        v19 = 259;
        sub_37011E0((unsigned __int64 *)v16, v5, (unsigned int *)(a4 + 6), v18[0].m128i_i64);
        v11 = v16[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          *a1 = 0;
          v16[0] = v11 | 1;
          sub_9C6670(a1, v16);
          sub_9C66B0(v16);
        }
        else
        {
          v16[0] = 0;
          sub_9C66B0(v16);
          sub_370D750(v18[0].m128i_i64, v5, (int **)(a4 + 16), (int **)(a4 + 32), (*(_WORD *)(a4 + 4) & 0x200) != 0);
          v12 = v18[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (v18[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            *a1 = 0;
            v18[0].m128i_i64[0] = v12 | 1;
            sub_9C6670(a1, v18);
          }
          else
          {
            v18[0].m128i_i64[0] = 0;
            sub_9C66B0(v18[0].m128i_i64);
            *a1 = 1;
          }
          sub_9C66B0(v18[0].m128i_i64);
        }
      }
    }
  }
  if ( (__int64 *)v14.m128i_i64[0] != &v15 )
    j_j___libc_free_0(v14.m128i_u64[0]);
  return a1;
}
