// Function: sub_10E8FE0
// Address: 0x10e8fe0
//
__int64 __fastcall sub_10E8FE0(__int64 a1, unsigned __int8 *a2)
{
  _BYTE v3[16]; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v4)(_BYTE *, __int64, int); // [rsp+10h] [rbp-20h]
  bool (__fastcall *v5)(__int64, __int64); // [rsp+18h] [rbp-18h]

  v5 = sub_10DF530;
  v4 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_10DF230;
  sub_10E8D80(a2, a1, (__int64)v3);
  if ( v4 )
    v4(v3, (__int64)v3, 3);
  return 0;
}
