// Function: sub_85EB10
// Address: 0x85eb10
//
__int64 __fastcall sub_85EB10(__int64 a1)
{
  char v1; // al
  _QWORD *i; // rax
  __int64 result; // rax
  int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rdx
  _BYTE *v7; // rdi

  v1 = *(_BYTE *)(a1 + 28);
  if ( v1 )
  {
    if ( v1 == 3 )
    {
      v7 = *(_BYTE **)(a1 + 32);
      if ( (v7[124] & 1) != 0 )
        return *(_QWORD *)(*(_QWORD *)sub_735B70((__int64)v7) + 96LL);
      else
        return *(_QWORD *)(*(_QWORD *)v7 + 96LL);
    }
    else
    {
      v4 = *(_DWORD *)(a1 + 240);
      if ( v4 == -1 )
      {
        return 0;
      }
      else
      {
        v5 = qword_4F04C68[0] + 776LL * v4;
        result = *(_QWORD *)(v5 + 24);
        v6 = v5 + 32;
        if ( !result )
          return v6;
      }
    }
  }
  else if ( qword_4F07288 == a1 )
  {
    return qword_4D03FF0 + 24;
  }
  else
  {
    for ( i = qword_4D03FD0; i[1] != a1; i = (_QWORD *)*i )
      ;
    return (__int64)(i + 3);
  }
  return result;
}
