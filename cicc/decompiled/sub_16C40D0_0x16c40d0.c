// Function: sub_16C40D0
// Address: 0x16c40d0
//
__int64 __fastcall sub_16C40D0(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax
  __int64 v4; // rdx

  result = sub_16C40A0(a1, a2, a3);
  do
  {
    if ( !v4 )
      break;
    --v4;
  }
  while ( *(_BYTE *)(result + v4) != 46 );
  return result;
}
