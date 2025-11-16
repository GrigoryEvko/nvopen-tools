// Function: sub_746720
// Address: 0x746720
//
__int64 __fastcall sub_746720(char a1, __int64 a2)
{
  int v3; // esi
  char *v4; // rax

  if ( *(_BYTE *)(a2 + 136) )
  {
    v3 = 1;
    if ( !*(_BYTE *)(a2 + 141) && dword_4F077C4 == 2 )
    {
      v3 = unk_4F068E0;
      if ( unk_4F068E0 )
      {
        if ( dword_4F077C0 )
        {
          v3 = 0;
          if ( !(_DWORD)qword_4F077B4 )
            v3 = qword_4F077A8 >= 0x11170u;
        }
        else
        {
          v3 = dword_4F077BC;
          if ( dword_4F077BC )
          {
            v3 = 0;
            if ( !(_DWORD)qword_4F077B4 )
              v3 = qword_4F077A8 > 0x1FBCFu;
          }
        }
      }
    }
  }
  else
  {
    v3 = dword_4F077C4 != 2;
  }
  v4 = sub_7465E0(a1, v3);
  return (*(__int64 (__fastcall **)(char *, __int64))a2)(v4, a2);
}
