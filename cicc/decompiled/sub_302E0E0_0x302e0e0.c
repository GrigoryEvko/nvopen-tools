// Function: sub_302E0E0
// Address: 0x302e0e0
//
__int64 __fastcall sub_302E0E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v6; // rax

  v2 = 0;
  v3 = **(_QWORD **)(a2 + 40);
  v4 = *(_QWORD *)(v3 + 56);
  if ( v4 )
  {
    if ( !*(_QWORD *)(v4 + 32) )
    {
      v2 = 1;
      if ( *(_DWORD *)(v3 + 24) == 213 )
      {
        v2 = 0;
        v6 = *(_QWORD *)(**(_QWORD **)(v3 + 40) + 56LL);
        if ( v6 )
          LOBYTE(v2) = *(_QWORD *)(v6 + 32) == 0;
      }
    }
  }
  return v2;
}
