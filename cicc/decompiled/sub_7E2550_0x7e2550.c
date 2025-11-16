// Function: sub_7E2550
// Address: 0x7e2550
//
__int64 __fastcall sub_7E2550(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx

  *a2 = 0;
  result = 0;
  if ( *(_BYTE *)(a1 + 24) == 1 && *(_BYTE *)(a1 + 56) == 73 )
  {
    v3 = *(_QWORD *)(a1 + 72);
    if ( *(_BYTE *)(v3 + 24) == 3 )
    {
      v4 = *(_QWORD *)(v3 + 56);
      if ( !*(_QWORD *)(v4 + 8) )
      {
        *a2 = v4;
        return 1;
      }
    }
  }
  return result;
}
