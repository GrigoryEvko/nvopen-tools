// Function: sub_1BBA0A0
// Address: 0x1bba0a0
//
void __fastcall sub_1BBA0A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  _QWORD *v4; // rsi
  _QWORD v5[3]; // [rsp+8h] [rbp-18h] BYREF

  v5[0] = a2;
  if ( a2 == *(_QWORD *)(a2 + 8) && !*(_DWORD *)(a2 + 96) && !*(_BYTE *)(a2 + 100) )
  {
    v2 = *a1;
    v4 = sub_1BB9480(*a1, (__int64)v5);
    if ( v3 )
      sub_1BB9520(v2, (__int64)v4, v3, (__int64)v5);
  }
}
