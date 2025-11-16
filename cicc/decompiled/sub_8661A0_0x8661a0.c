// Function: sub_8661A0
// Address: 0x8661a0
//
__int64 __fastcall sub_8661A0(int a1)
{
  int v1; // eax
  __int64 v2; // rax
  char v3; // dl
  __int64 v4; // rcx

  if ( !a1 )
    return *(_QWORD *)(sub_85B130() + 408);
  v1 = unk_4F04C48;
  if ( unk_4F04C48 != -1 )
  {
    v2 = qword_4F04C68[0] + 776LL * unk_4F04C48;
    if ( *(_BYTE *)(v2 + 4) == 9 )
    {
LABEL_4:
      while ( (*(_BYTE *)(v2 + 12) & 0x10) != 0 )
      {
        v3 = *(_BYTE *)(v2 - 772);
        v4 = v2 - 776;
        if ( v3 && (v2 -= 776, v3 != 9) )
        {
          while ( 1 )
          {
            v3 = *(_BYTE *)(v2 - 772);
            v2 -= 776;
            if ( !v3 )
              break;
            if ( v3 == 9 )
              goto LABEL_4;
          }
        }
        else
        {
          v2 = v4;
        }
        if ( v3 != 9 )
          goto LABEL_11;
      }
      v1 = 1594008481 * ((v2 - qword_4F04C68[0]) >> 3);
    }
    else
    {
LABEL_11:
      v1 = -1;
    }
  }
  if ( dword_4F04C44 >= v1 )
    v1 = dword_4F04C44;
  return *(_QWORD *)(qword_4F04C68[0] + 776LL * v1 + 408);
}
