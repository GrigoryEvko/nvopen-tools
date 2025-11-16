// Function: sub_688510
// Address: 0x688510
//
__int64 __fastcall sub_688510(__int64 a1, _QWORD *a2, _DWORD *a3)
{
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rax

  if ( (dword_4F04C44 != -1
     || (v4 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v4 + 6) & 6) != 0)
     || *(_BYTE *)(v4 + 4) == 12)
    && (unsigned int)sub_8DBE70(*a2) )
  {
    *a3 = 1;
    return 0;
  }
  else
  {
    v5 = *(_BYTE *)(*a2 + 140LL);
    if ( v5 == 12 )
    {
      v6 = *a2;
      do
      {
        v6 = *(_QWORD *)(v6 + 160);
        v5 = *(_BYTE *)(v6 + 140);
      }
      while ( v5 == 12 );
    }
    if ( !v5 )
      goto LABEL_8;
    if ( !(unsigned int)sub_8D2B80(*a2) )
    {
      if ( a1 )
        *(_BYTE *)(a1 + 56) = 1;
      else
        sub_6851C0(0x9D7u, (_DWORD *)a2 + 17);
LABEL_8:
      *a3 = 0;
      return 0;
    }
    *a3 = 0;
    return 1;
  }
}
