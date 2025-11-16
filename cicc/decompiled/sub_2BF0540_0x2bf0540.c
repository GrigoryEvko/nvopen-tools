// Function: sub_2BF0540
// Address: 0x2bf0540
//
__int64 __fastcall sub_2BF0540(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = a1;
  if ( !*(_DWORD *)(a1 + 88) )
  {
    do
    {
      v2 = *(_QWORD *)(result + 48);
      if ( !v2 )
        break;
      result = *(_QWORD *)(result + 48);
    }
    while ( !*(_DWORD *)(v2 + 88) );
  }
  return result;
}
