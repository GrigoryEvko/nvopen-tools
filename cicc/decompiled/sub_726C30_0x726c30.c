// Function: sub_726C30
// Address: 0x726c30
//
_BYTE *__fastcall sub_726C30(char a1)
{
  __int64 v1; // rdx
  __int64 v2; // rdx
  _BYTE *result; // rax

  if ( dword_4F07270[0] == unk_4F073B8 )
  {
    v1 = 0;
  }
  else
  {
    if ( dword_4F04C58 == -1 )
    {
LABEL_7:
      result = sub_7246D0(64);
      goto LABEL_6;
    }
    v1 = 776LL * dword_4F04C58;
  }
  v2 = qword_4F04C68[0] + v1;
  result = *(_BYTE **)(v2 + 504);
  if ( !result )
    goto LABEL_7;
  *(_QWORD *)(v2 + 504) = *((_QWORD *)result + 7);
LABEL_6:
  result[1] &= 0xF8u;
  result[8] = 0;
  *((_QWORD *)result + 2) = 0;
  *result = a1;
  *((_QWORD *)result + 3) = 0;
  *((_QWORD *)result + 4) = 0;
  *((_QWORD *)result + 5) = 0;
  *((_QWORD *)result + 6) = 0;
  *((_QWORD *)result + 7) = 0;
  return result;
}
