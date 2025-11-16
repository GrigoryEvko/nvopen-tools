// Function: sub_6E8430
// Address: 0x6e8430
//
__int64 __fastcall sub_6E8430(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 v3; // rbx

  v1 = a1;
  if ( *(_BYTE *)(a1 + 24) == 1 )
  {
    do
    {
      if ( *(_BYTE *)(v1 + 56) != 5 )
        break;
      v3 = *(_QWORD *)(v1 + 72);
      if ( dword_4F04C44 != -1
        || (v2 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v2 + 6) & 6) != 0)
        || *(_BYTE *)(v2 + 4) == 12 )
      {
        if ( (unsigned int)sub_8DD3B0(*(_QWORD *)v1) || (unsigned int)sub_8DD3B0(*(_QWORD *)v3) )
          break;
      }
      v1 = v3;
    }
    while ( *(_BYTE *)(v3 + 24) == 1 );
  }
  return v1;
}
