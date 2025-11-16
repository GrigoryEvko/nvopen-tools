// Function: sub_8809D0
// Address: 0x8809d0
//
__int64 __fastcall sub_8809D0(__int64 a1)
{
  unsigned int v1; // r8d
  int v3; // ecx
  __int64 v4; // rax
  __int64 v5; // rax

  v1 = 0;
  if ( (*(_BYTE *)(a1 + 81) & 0x10) == 0 && !*(_QWORD *)(a1 + 64) )
  {
    v3 = *(_DWORD *)(a1 + 40);
    if ( v3 != unk_4F066A8 && (*(_BYTE *)(a1 + 82) & 8) == 0 )
    {
      v4 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( v4 )
      {
        do
        {
          if ( v3 == *(_DWORD *)v4 )
            return (*(_BYTE *)(v4 + 4) == 17) | (unsigned __int8)(*(_BYTE *)(v4 + 4) == 2);
          v5 = *(int *)(v4 + 552);
          if ( (_DWORD)v5 == -1 )
            break;
          v4 = qword_4F04C68[0] + 776 * v5;
        }
        while ( v4 );
        return 0;
      }
    }
  }
  return v1;
}
