// Function: sub_1F44020
// Address: 0x1f44020
//
_QWORD *__fastcall sub_1F44020(__int64 a1, __int64 a2, char a3)
{
  _QWORD **v4; // r13
  __int64 v5; // rax
  _QWORD *v6; // rdi
  _QWORD *v7; // r12
  __int64 v9; // r15
  __int16 v10; // bx
  __m128i v11; // [rsp+0h] [rbp-110h] BYREF
  char v12; // [rsp+10h] [rbp-100h]
  char v13; // [rsp+11h] [rbp-FFh]
  __m128i v14; // [rsp+20h] [rbp-F0h] BYREF
  char v15; // [rsp+30h] [rbp-E0h]
  char v16; // [rsp+31h] [rbp-DFh]
  __m128i v17[2]; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v18; // [rsp+60h] [rbp-B0h] BYREF
  __int16 v19; // [rsp+70h] [rbp-A0h]
  __m128i v20; // [rsp+80h] [rbp-90h] BYREF
  char v21; // [rsp+90h] [rbp-80h]
  char v22; // [rsp+91h] [rbp-7Fh]
  __m128i v23; // [rsp+A0h] [rbp-70h] BYREF
  char v24; // [rsp+B0h] [rbp-60h]
  char v25; // [rsp+B1h] [rbp-5Fh]
  __m128i v26; // [rsp+C0h] [rbp-50h] BYREF
  char v27; // [rsp+D0h] [rbp-40h]
  char v28; // [rsp+D1h] [rbp-3Fh]

  v4 = *(_QWORD ***)(*(_QWORD *)(*(_QWORD *)(a2 + 8) + 56LL) + 40LL);
  v5 = sub_1632000((__int64)v4, (__int64)"__safestack_unsafe_stack_ptr", 28);
  if ( v5 )
  {
    v6 = *v4;
    v7 = (_QWORD *)v5;
    if ( *(_BYTE *)(v5 + 16) == 3 )
    {
      if ( *(_QWORD *)(v5 + 24) == sub_16471D0(v6, 0) )
      {
        if ( a3 == ((*((_BYTE *)v7 + 33) & 0x1C) != 0) )
          return v7;
        v25 = 1;
        v23.m128i_i64[0] = (__int64)"be thread-local";
        v24 = 3;
        if ( a3 )
        {
          v19 = 257;
        }
        else
        {
          v18.m128i_i64[0] = (__int64)"not ";
          v19 = 259;
        }
        v16 = 1;
        v14.m128i_i64[0] = (__int64)" must ";
        v15 = 3;
        v13 = 1;
        v11.m128i_i64[0] = (__int64)"__safestack_unsafe_stack_ptr";
        v12 = 3;
        sub_14EC200(v17, &v11, &v14);
        sub_14EC200(&v20, v17, &v18);
      }
      else
      {
        v25 = 1;
        v23.m128i_i64[0] = (__int64)" must have void* type";
        v24 = 3;
        v22 = 1;
        v20.m128i_i64[0] = (__int64)"__safestack_unsafe_stack_ptr";
        v21 = 3;
      }
      sub_14EC200(&v26, &v20, &v23);
      sub_16BCFB0((__int64)&v26, 1u);
    }
    v9 = sub_16471D0(v6, 0);
  }
  else
  {
    v9 = sub_16471D0(*v4, 0);
  }
  v28 = 1;
  v26.m128i_i64[0] = (__int64)"__safestack_unsafe_stack_ptr";
  v27 = 3;
  v10 = a3 != 0 ? 3 : 0;
  v7 = sub_1648A60(88, 1u);
  if ( v7 )
    sub_15E51E0((__int64)v7, (__int64)v4, v9, 0, 0, 0, (__int64)&v26, 0, v10, 0, 0);
  return v7;
}
