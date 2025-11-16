// Function: sub_3154C60
// Address: 0x3154c60
//
char __fastcall sub_3154C60(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  char result; // al
  _DWORD v5[3]; // [rsp+Ch] [rbp-14h] BYREF

  v5[0] = 0;
  result = sub_3154990(a1, v5, a3, a4);
  if ( result )
    return v5[0] == a2;
  return result;
}
