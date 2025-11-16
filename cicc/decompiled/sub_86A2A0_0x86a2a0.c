// Function: sub_86A2A0
// Address: 0x86a2a0
//
__int64 __fastcall sub_86A2A0(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // rax
  __int64 v3; // rdx

  v1 = 0;
  if ( !dword_4F04C3C )
  {
    v2 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 336);
    if ( v2 )
    {
      while ( 1 )
      {
        v1 = v2;
        v2 = *(_QWORD *)(v2 + 8);
        v3 = *(_QWORD *)(v1 + 24);
        if ( v3 == a1 || *(_BYTE *)(v1 + 16) == 53 && *(_QWORD *)(v3 + 24) == a1 )
          break;
        if ( !v2 )
          return 0;
      }
    }
  }
  return v1;
}
