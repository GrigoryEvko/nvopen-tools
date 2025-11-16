// Function: sub_679AE0
// Address: 0x679ae0
//
__int64 *__fastcall sub_679AE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 *result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // r12d
  unsigned int v11; // r15d

  v4 = a1;
  sub_7B8B50(a1, a2, a3, a4);
  result = (__int64 *)word_4F06418[0];
  if ( word_4F06418[0] != 27 )
    goto LABEL_2;
  sub_7B8B50(a1, a2, v5, v6);
  v10 = 0;
  v11 = (a1 & 0x40) == 0 ? 16385 : 17409;
LABEL_5:
  while ( 2 )
  {
    sub_7B8B50(a1, a2, v8, v9);
    if ( dword_4F077C4 != 2 )
    {
LABEL_6:
      result = (__int64 *)word_4F06418[0];
      if ( word_4F06418[0] == 28 )
        goto LABEL_13;
      goto LABEL_7;
    }
    while ( 1 )
    {
      if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 )
        goto LABEL_5;
      a2 = 0;
      a1 = v11;
      sub_7C0F00(v11, 0);
      result = (__int64 *)word_4F06418[0];
      if ( word_4F06418[0] == 28 )
        break;
LABEL_7:
      if ( (_WORD)result == 27 )
      {
        ++v10;
        goto LABEL_5;
      }
      if ( ((unsigned __int16)result & 0xFFBF) == 9 || (_WORD)result == 75 )
        return result;
      sub_7B8B50(a1, a2, (unsigned int)result & 0xFFFFFFBF, v9);
      if ( dword_4F077C4 != 2 )
        goto LABEL_6;
    }
LABEL_13:
    if ( v10 )
    {
      --v10;
      continue;
    }
    break;
  }
  sub_679930(v4, 0, v8, v9);
  result = (__int64 *)word_4F06418[0];
LABEL_2:
  if ( (_WORD)result == 28 )
    return sub_679930(v4, 0, v5, v6);
  return result;
}
