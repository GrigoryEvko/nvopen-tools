// Function: sub_2AB77E0
// Address: 0x2ab77e0
//
__int64 __fastcall sub_2AB77E0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  _QWORD v5[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v6)(const __m128i **, const __m128i *, int); // [rsp+10h] [rbp-20h]
  __int64 (__fastcall *v7)(); // [rsp+18h] [rbp-18h]

  v5[0] = a1;
  v7 = sub_2AC3980;
  v5[1] = a2;
  v6 = sub_2AA7D50;
  v3 = sub_2BF1270(v5, a3) ^ 1;
  if ( v6 )
    v6((const __m128i **)v5, (const __m128i *)v5, 3);
  return v3;
}
