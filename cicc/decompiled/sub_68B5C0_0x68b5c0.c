// Function: sub_68B5C0
// Address: 0x68b5c0
//
__int64 __fastcall sub_68B5C0(_DWORD *a1, _DWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  char v5; // dl
  __int64 v6; // rax

  if ( (dword_4F04C44 != -1
     || (v3 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v3 + 6) & 6) != 0)
     || *(_BYTE *)(v3 + 4) == 12)
    && (unsigned int)sub_8DBE70(*(_QWORD *)a1) )
  {
    *a2 = 1;
    return 0;
  }
  else
  {
    v4 = *(_QWORD *)a1;
    v5 = *(_BYTE *)(*(_QWORD *)a1 + 140LL);
    if ( v5 == 12 )
    {
      v6 = *(_QWORD *)a1;
      do
      {
        v6 = *(_QWORD *)(v6 + 160);
        v5 = *(_BYTE *)(v6 + 140);
      }
      while ( v5 == 12 );
    }
    if ( v5 )
    {
      if ( (unsigned int)sub_8D2E30(v4) )
      {
        *a2 = 0;
        return 1;
      }
      else
      {
        sub_6851C0(0xD82u, a1 + 17);
        *a2 = 0;
        return 0;
      }
    }
    else
    {
      *a2 = 0;
      return 0;
    }
  }
}
