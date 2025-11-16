// Function: sub_87E070
// Address: 0x87e070
//
_BOOL8 __fastcall sub_87E070(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  bool v3; // al
  _BOOL4 v4; // r8d
  __int64 v5; // rbx
  __int64 v6; // rdi
  int v7; // eax

  v2 = (int)dword_4F04C2C;
  if ( dword_4F04C2C != -1 )
  {
    while ( 1 )
    {
      v5 = qword_4F04C68[0] + 776 * v2;
      if ( (unsigned __int8)(*(_BYTE *)(v5 + 4) - 6) <= 1u
        && (v6 = sub_8D5CE0(*(_QWORD *)(v5 + 208), *(_QWORD *)(a2 + 40))) != 0 )
      {
        v7 = sub_87DF20(v6);
        v2 = *(int *)(v5 + 448);
        v3 = v7 != 0;
        v4 = v3;
        if ( (_DWORD)v2 == -1 )
          return v4;
      }
      else
      {
        v2 = *(int *)(v5 + 448);
        v3 = 0;
        v4 = 0;
        if ( (_DWORD)v2 == -1 )
          return v4;
      }
      if ( v3 )
        return v4;
    }
  }
  return 0;
}
