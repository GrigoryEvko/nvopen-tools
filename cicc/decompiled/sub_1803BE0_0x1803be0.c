// Function: sub_1803BE0
// Address: 0x1803be0
//
bool __fastcall sub_1803BE0(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4)
{
  bool v6; // cf
  bool result; // al
  unsigned int v8; // [rsp+4h] [rbp-2Ch] BYREF
  unsigned int v9; // [rsp+8h] [rbp-28h] BYREF
  _DWORD v10[9]; // [rsp+Ch] [rbp-24h] BYREF

  sub_16E2390(a1, &v8, &v9, v10);
  v6 = v8 < a2;
  if ( v8 != a2 )
    return v6;
  v6 = v9 < a3;
  if ( v9 != a3 )
    return v6;
  result = 0;
  if ( v10[0] != a4 )
    return v9 < a4;
  return result;
}
