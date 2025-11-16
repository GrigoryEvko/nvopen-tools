// Function: sub_867B10
// Address: 0x867b10
//
__int64 sub_867B10()
{
  __int64 result; // rax
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // rax
  char v4; // al
  bool v5; // zf
  int v6; // eax

  if ( !qword_4F04C50 )
  {
    v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( *(_BYTE *)(v1 + 4) == 9 )
    {
      v2 = 0;
      while ( 1 )
      {
        v3 = *(_QWORD *)(v1 + 360);
        if ( !v3 || *(_BYTE *)(v3 + 80) != 3 )
          break;
LABEL_7:
        v4 = *(_BYTE *)(v1 - 772);
        while ( (unsigned __int8)(v4 - 4) <= 1u || v4 == 7 )
        {
          v4 = *(_BYTE *)(v1 - 1548);
          if ( v4 == 9 )
          {
            v1 -= 1552;
            goto LABEL_7;
          }
          v1 -= 776;
        }
        v5 = *(_BYTE *)(v1 - 1548) == 9;
        v2 = 1;
        v1 -= 1552;
        if ( !v5 )
          goto LABEL_14;
      }
      if ( v2 )
        goto LABEL_14;
    }
    if ( dword_4F04C38 )
    {
LABEL_14:
      while ( qword_4F04C68[0] != v1 )
      {
        result = *(_QWORD *)(v1 + 184);
        if ( result && *(_BYTE *)(result + 28) == 17 )
          return result;
        v6 = *(_DWORD *)(v1 + 552);
        v1 = 0;
        if ( v6 != -1 )
          v1 = qword_4F04C68[0] + 776LL * v6;
      }
    }
  }
  return qword_4F04C50;
}
