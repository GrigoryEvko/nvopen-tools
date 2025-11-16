// Function: sub_812380
// Address: 0x812380
//
_QWORD *__fastcall sub_812380(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  const char *v6; // r14
  size_t v7; // rax
  __int64 v8; // rdx
  size_t v9; // rax
  _QWORD *result; // rax
  _QWORD v11[14]; // [rsp+0h] [rbp-70h] BYREF

  if ( (*(_BYTE *)(a1 + 89) & 8) != 0 )
  {
    v6 = *(const char **)(a1 + 24);
    v7 = strlen(v6);
    if ( v7 <= 9 )
    {
LABEL_3:
      v8 = 1;
      LOWORD(v11[0]) = (unsigned __int8)(v7 + 48);
      goto LABEL_4;
    }
  }
  else
  {
    v6 = *(const char **)(a1 + 8);
    v7 = strlen(v6);
    if ( v7 <= 9 )
      goto LABEL_3;
  }
  v8 = (int)sub_622470(v7, v11);
LABEL_4:
  *a4 += v8;
  sub_8238B0(qword_4F18BE0, v11, v8);
  v9 = strlen(v6);
  *a4 += v9;
  result = (_QWORD *)sub_8238B0(qword_4F18BE0, v6, v9);
  if ( a3 )
  {
    if ( (*(_BYTE *)(a3 + 33) & 2) == 0 )
      return result;
LABEL_10:
    v11[0] = a2;
    return sub_811CB0(v11, a3, 0, a4);
  }
  if ( a2 )
    goto LABEL_10;
  return result;
}
