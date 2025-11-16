// Function: sub_E45AA0
// Address: 0xe45aa0
//
__int64 __fastcall sub_E45AA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v6; // zf
  _QWORD v8[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v9[3]; // [rsp+10h] [rbp-20h] BYREF

  v6 = *(_QWORD *)(a2 + 16) == 0;
  v8[0] = a3;
  v8[1] = a4;
  v9[0] = a5;
  v9[1] = a6;
  if ( v6 )
    sub_4263D6(a1, a2, a3);
  (*(void (__fastcall **)(__int64, __int64, _QWORD *, _QWORD *))(a2 + 24))(a1, a2, v8, v9);
  return a1;
}
