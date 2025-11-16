// Function: sub_85E680
// Address: 0x85e680
//
__int64 __fastcall sub_85E680(__int64 a1, int a2)
{
  __int64 v2; // rax
  char v4; // dl
  __int64 v5; // r12
  __int64 result; // rax
  _QWORD *v7; // rax
  _QWORD *i; // rdi
  unsigned __int8 v9; // al

  v2 = qword_4F04C68[0] + 776LL * a2;
  v4 = *(_BYTE *)(v2 + 4);
  v5 = *(_QWORD *)(a1 + 88);
  if ( ((v4 - 15) & 0xFD) == 0 || v4 == 2 )
  {
    v7 = (_QWORD *)sub_85B7C0((_QWORD *)a1, a2);
    for ( i = (_QWORD *)*v7; i[1] != a1; i = (_QWORD *)*i )
      v7 = i;
    *v7 = *i;
    *i = 0;
    sub_878490(i);
    v9 = *(_BYTE *)(a1 + 80);
    if ( v9 <= 5u )
    {
      if ( v9 > 3u )
        goto LABEL_8;
    }
    else if ( v9 == 6 )
    {
      goto LABEL_13;
    }
    sub_721090();
  }
  --*(_QWORD *)(v2 + 688);
  result = *(unsigned __int8 *)(a1 + 80);
  if ( (unsigned __int8)(result - 4) > 1u )
  {
    if ( (_BYTE)result != 6 )
      return result;
LABEL_13:
    result = *(_QWORD *)(a1 + 96);
    *(_QWORD *)(result + 16) = 0;
    return result;
  }
  result = *(_QWORD *)(a1 + 88);
  if ( result && (*(_BYTE *)(result + 177) & 0x20) == 0 )
  {
LABEL_8:
    result = *(_QWORD *)(a1 + 96);
    *(_QWORD *)(result + 168) = *(_BYTE *)(*(_QWORD *)(v5 + 168) + 113LL) != 0;
  }
  return result;
}
