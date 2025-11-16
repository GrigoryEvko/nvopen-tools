// Function: sub_1263BC0
// Address: 0x1263bc0
//
__int64 __fastcall sub_1263BC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = a1;
  do
  {
    if ( !a2 )
      break;
    --a2;
  }
  while ( *(_BYTE *)(a1 + a2) != 46 );
  return result;
}
