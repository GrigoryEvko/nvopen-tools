// Function: sub_310A360
// Address: 0x310a360
//
__int64 *__fastcall sub_310A360(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        const __m128i *a6,
        char a7,
        const char **a8)
{
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v17; // [rsp-8h] [rbp-58h]
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v10 = *a5;
  *a5 = 0;
  v20[0] = v10;
  v11 = sub_22077B0(0xB8u);
  v13 = v11;
  if ( v11 )
  {
    sub_31099F0(v11, a2, a3, a4, v20, a6, a7, a8);
    v12 = v17;
  }
  if ( v20[0] )
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v20[0] + 8LL))(v20[0], a2, v12);
  if ( !*(_BYTE *)(v13 + 88) )
  {
    v14 = *(_QWORD *)v13;
    v15 = v13;
    v13 = 0;
    (*(void (__fastcall **)(__int64, __int64, __int64))(v14 + 8))(v15, a2, v12);
  }
  *a1 = v13;
  return a1;
}
