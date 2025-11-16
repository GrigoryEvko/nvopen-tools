// Function: sub_228EC60
// Address: 0x228ec60
//
char __fastcall sub_228EC60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v6; // r8
  char result; // al

  v6 = sub_228DFC0(a1, 0x20u, a2, a3);
  result = 0;
  if ( !v6 )
  {
    result = sub_228DFC0(a1, 0x21u, a2, a3);
    if ( !result )
      *(_BYTE *)(a4 + 43) = 0;
  }
  return result;
}
