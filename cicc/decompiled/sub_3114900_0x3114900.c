// Function: sub_3114900
// Address: 0x3114900
//
__int64 __fastcall sub_3114900(unsigned __int64 *a1, char a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v7; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v8; // [rsp+10h] [rbp-50h] BYREF
  char v9; // [rsp+18h] [rbp-48h]
  __int64 (__fastcall *v10)(const __m128i **, const __m128i *, int); // [rsp+20h] [rbp-40h]
  __int64 (__fastcall *v11)(__int64, __int64); // [rsp+28h] [rbp-38h]
  _BYTE v12[16]; // [rsp+30h] [rbp-30h] BYREF
  void (__fastcall *v13)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-20h]

  v8 = &v7;
  v9 = a2;
  v11 = sub_3113B90;
  v10 = sub_3113BC0;
  v7 = 0;
  v13 = 0;
  sub_31144C0(a1, (__int64)&v8, (__int64)v12, 0, a5, a6);
  if ( v10 )
    v10((const __m128i **)&v8, (const __m128i *)&v8, 3);
  if ( v13 )
    v13(v12, v12, 3);
  return v7;
}
