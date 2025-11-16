// Function: sub_1BE23D0
// Address: 0x1be23d0
//
__int64 __fastcall sub_1BE23D0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = a1;
  if ( !*(_DWORD *)(a1 + 64) )
  {
    do
    {
      v2 = *(_QWORD *)(result + 48);
      if ( !v2 )
        break;
      result = *(_QWORD *)(result + 48);
    }
    while ( !*(_DWORD *)(v2 + 64) );
  }
  return result;
}
