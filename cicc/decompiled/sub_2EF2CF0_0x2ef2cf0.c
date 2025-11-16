// Function: sub_2EF2CF0
// Address: 0x2ef2cf0
//
int __fastcall sub_2EF2CF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int result; // eax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i v10[3]; // [rsp-108h] [rbp-108h] BYREF
  __m128i v11[3]; // [rsp-D8h] [rbp-D8h] BYREF
  __m128i v12; // [rsp-A8h] [rbp-A8h] BYREF
  char v13; // [rsp-88h] [rbp-88h]
  char v14; // [rsp-87h] [rbp-87h]
  __m128i v15; // [rsp-78h] [rbp-78h] BYREF
  __int16 v16; // [rsp-58h] [rbp-58h]
  __m128i v17; // [rsp-48h] [rbp-48h] BYREF
  char v18; // [rsp-28h] [rbp-28h]
  char v19; // [rsp-27h] [rbp-27h]

  result = *(_DWORD *)a1;
  if ( *(_DWORD *)a1 )
  {
    if ( *(_BYTE *)(a1 + 4) )
    {
      v15.m128i_i32[0] = *(_DWORD *)a1;
      v17.m128i_i64[0] = (__int64)" machine code errors.";
      v12.m128i_i64[0] = (__int64)"Found ";
      v19 = 1;
      v18 = 3;
      v16 = 265;
      v14 = 1;
      v13 = 3;
      sub_9C6370(v11, &v12, &v15, (__int64)" machine code errors.", a5, a6);
      sub_9C6370(v10, v11, &v17, v7, v8, v9);
      sub_C64D30((__int64)v10, 1u);
    }
    result = (int)qword_5022360;
    if ( !qword_5022360 )
      result = sub_C7D570((__int64 *)&qword_5022360, (__int64 (*)(void))sub_BC3580, (__int64)sub_BC3540);
    if ( &_pthread_key_create )
      return pthread_mutex_unlock(qword_5022360);
  }
  return result;
}
