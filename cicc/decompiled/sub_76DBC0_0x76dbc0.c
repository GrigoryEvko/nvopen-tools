// Function: sub_76DBC0
// Address: 0x76dbc0
//
__int64 __fastcall sub_76DBC0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  int v3; // ecx

  v1 = *(_QWORD *)(a1 + 264);
  result = a1;
  if ( v1 )
  {
    v3 = *(_DWORD *)(v1 + 16);
    if ( v3 )
    {
      result = 0;
      if ( v3 == 1 )
      {
        result = *(_QWORD *)(v1 + 24);
        *(_WORD *)(v1 + 56) = 257;
      }
    }
  }
  return result;
}
