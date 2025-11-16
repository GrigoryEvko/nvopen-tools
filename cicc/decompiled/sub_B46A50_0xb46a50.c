// Function: sub_B46A50
// Address: 0xb46a50
//
__int64 __fastcall sub_B46A50(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  int v3; // edx

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
      {
        v3 = *(_DWORD *)(v2 + 36);
        LOBYTE(result) = v3 == 208;
        LOBYTE(v3) = v3 == 346;
        return v3 | (unsigned int)result;
      }
    }
  }
  return result;
}
