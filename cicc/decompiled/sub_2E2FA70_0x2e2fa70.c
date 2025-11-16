// Function: sub_2E2FA70
// Address: 0x2e2fa70
//
__int64 __fastcall sub_2E2FA70(_BYTE *a1)
{
  __int64 result; // rax

  result = 0;
  if ( !*a1 )
    return ((a1[3] >> 4) ^ 1) & 1;
  return result;
}
