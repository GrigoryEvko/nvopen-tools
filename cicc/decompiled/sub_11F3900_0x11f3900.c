// Function: sub_11F3900
// Address: 0x11f3900
//
__int64 __fastcall sub_11F3900(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rdx
  char *v4; // rax
  __int64 v5; // rdx
  __m128i *v6; // rax
  __int64 v7; // rax
  __m128i *v9; // [rsp+0h] [rbp-90h] BYREF
  __int64 v10; // [rsp+8h] [rbp-88h]
  __m128i v11; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v12[14]; // [rsp+20h] [rbp-70h] BYREF

  v2 = a1 + 16;
  sub_BD5D20((__int64)a2);
  if ( v3 )
  {
    v4 = (char *)sub_BD5D20((__int64)a2);
    *(_QWORD *)a1 = v2;
    if ( v4 )
    {
      sub_11F33F0((__int64 *)a1, v4, (__int64)&v4[v5]);
    }
    else
    {
      *(_QWORD *)(a1 + 8) = 0;
      *(_BYTE *)(a1 + 16) = 0;
    }
  }
  else
  {
    v11.m128i_i8[0] = 0;
    v9 = &v11;
    v12[5] = 0x100000000LL;
    v12[6] = &v9;
    v10 = 0;
    v12[0] = &unk_49DD210;
    memset(&v12[1], 0, 32);
    sub_CB5980((__int64)v12, 0, 0, 0);
    sub_A5BF40(a2, (__int64)v12, 0, 0);
    v6 = v9;
    *(_QWORD *)a1 = v2;
    if ( v6 == &v11 )
    {
      *(__m128i *)(a1 + 16) = _mm_load_si128(&v11);
    }
    else
    {
      *(_QWORD *)a1 = v6;
      *(_QWORD *)(a1 + 16) = v11.m128i_i64[0];
    }
    v7 = v10;
    v9 = &v11;
    v10 = 0;
    *(_QWORD *)(a1 + 8) = v7;
    v11.m128i_i8[0] = 0;
    v12[0] = &unk_49DD210;
    sub_CB5840((__int64)v12);
    if ( v9 != &v11 )
      j_j___libc_free_0(v9, v11.m128i_i64[0] + 1);
  }
  return a1;
}
