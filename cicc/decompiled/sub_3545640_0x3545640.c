// Function: sub_3545640
// Address: 0x3545640
//
__int64 __fastcall sub_3545640(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d

  v2 = (*(__int64 *)(a1 + 8) >> 1) & 3;
  if ( v2 == 3 && *(_DWORD *)(a1 + 16) == 3 )
    return 1;
  v3 = 1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 200LL) != -1 )
  {
    v3 = 0;
    if ( (_BYTE)a2 )
    {
      v3 = a2;
      if ( v2 != 1 )
        LOBYTE(v3) = *(_DWORD *)(a1 + 24) != 0;
    }
  }
  return v3;
}
