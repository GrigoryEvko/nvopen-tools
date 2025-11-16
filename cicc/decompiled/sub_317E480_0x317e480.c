// Function: sub_317E480
// Address: 0x317e480
//
__int64 __fastcall sub_317E480(__int64 a1, _DWORD *a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  unsigned __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // rax

  v2 = a1 + 8;
  v3 = *(_QWORD *)(a1 + 24);
  if ( v3 == a1 + 8 )
    return 0;
  v4 = 0;
  v5 = 0;
  do
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(v3 + 128) == *a2 && *(_DWORD *)(v3 + 132) == a2[1] )
      {
        v6 = sub_317E470(v3 + 40);
        if ( v6 )
        {
          v7 = *(_QWORD *)(v6 + 56);
          if ( v7 > v4 )
            break;
        }
      }
      v3 = sub_220EEE0(v3);
      if ( v2 == v3 )
        return v5;
    }
    v4 = v7;
    v5 = v3 + 40;
    v3 = sub_220EEE0(v3);
  }
  while ( v2 != v3 );
  return v5;
}
