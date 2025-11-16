// Function: sub_2FEF600
// Address: 0x2fef600
//
__int64 __fastcall sub_2FEF600(__int128 a1)
{
  __int128 *v1; // rax
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __m128i v9; // [rsp-118h] [rbp-118h] BYREF
  __int16 v10; // [rsp-F8h] [rbp-F8h]
  __int128 v11; // [rsp-E8h] [rbp-E8h] BYREF
  __int16 v12; // [rsp-C8h] [rbp-C8h]
  __m128i v13[3]; // [rsp-B8h] [rbp-B8h] BYREF
  __m128i v14; // [rsp-88h] [rbp-88h] BYREF
  char v15; // [rsp-68h] [rbp-68h]
  char v16; // [rsp-67h] [rbp-67h]
  __m128i v17[5]; // [rsp-58h] [rbp-58h] BYREF

  if ( !*((_QWORD *)&a1 + 1) )
    return 0;
  v1 = sub_BC2B00();
  result = sub_BC2D00((pthread_rwlock_t *)v1, a1, *((__int64 *)&a1 + 1));
  if ( !result )
  {
    v11 = a1;
    v14.m128i_i64[0] = (__int64)"\" pass is not registered.";
    v16 = 1;
    v15 = 3;
    v12 = 261;
    v9.m128i_i64[0] = 34;
    v10 = 264;
    sub_9C6370(v13, &v9, (const __m128i *)&v11, v3, v4, v5);
    sub_9C6370(v17, v13, &v14, v6, v7, v8);
    sub_C64D30((__int64)v17, 1u);
  }
  return result;
}
