// Function: sub_BE7A40
// Address: 0xbe7a40
//
void __fastcall sub_BE7A40(_BYTE *a1, __int64 a2, const void *a3, size_t a4, _BYTE *a5)
{
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // [rsp+8h] [rbp-108h]
  _BYTE *v12; // [rsp+10h] [rbp-100h] BYREF
  __int64 v13; // [rsp+18h] [rbp-F8h] BYREF
  _QWORD v14[4]; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v15; // [rsp+40h] [rbp-D0h]
  __m128i v16; // [rsp+50h] [rbp-C0h] BYREF
  const char *v17; // [rsp+60h] [rbp-B0h]
  __int16 v18; // [rsp+70h] [rbp-A0h]
  __m128i v19; // [rsp+80h] [rbp-90h] BYREF
  __int16 v20; // [rsp+A0h] [rbp-70h]
  __m128i v21[6]; // [rsp+B0h] [rbp-60h] BYREF

  v13 = a2;
  v12 = a5;
  if ( (unsigned __int8)sub_A747A0(&v13, a3, a4) )
  {
    v21[0].m128i_i64[0] = sub_A747B0(&v13, -1, a3, a4);
    v7 = sub_A72240(v21[0].m128i_i64);
    v11 = v8;
    if ( (unsigned __int8)sub_C93C90(v7, v8, 10, v21) || v21[0].m128i_i64[0] != v21[0].m128i_u32[0] )
    {
      v20 = 261;
      v14[0] = "\"";
      v19.m128i_i64[1] = v11;
      v15 = 1283;
      v16.m128i_i64[0] = (__int64)v14;
      v19.m128i_i64[0] = v7;
      v14[2] = a3;
      v14[3] = a4;
      v17 = "\" takes an unsigned integer: ";
      v18 = 770;
      sub_9C6370(v21, &v16, &v19, 770, v9, v10);
      sub_BE7760(a1, (__int64)v21, &v12);
    }
  }
}
