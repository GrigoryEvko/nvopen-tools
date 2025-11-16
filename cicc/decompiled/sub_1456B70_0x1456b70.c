// Function: sub_1456B70
// Address: 0x1456b70
//
_BYTE *__fastcall sub_1456B70(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r13
  __int64 v4; // rax
  void *v5; // rdx
  int v6; // eax
  _BYTE *result; // rax

  v3 = sub_16E8750(a2, a3);
  v4 = sub_1452540(a1);
  sub_1456620(v4, v3);
  v5 = *(void **)(v3 + 24);
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v5 <= 0xDu )
  {
    sub_16E7EE0(v3, " Added Flags: ", 14);
    v6 = *(_DWORD *)(a1 + 48);
    if ( (v6 & 1) == 0 )
    {
LABEL_3:
      if ( (v6 & 2) == 0 )
        goto LABEL_4;
LABEL_8:
      sub_1263B40(a2, "<nssw>");
      result = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) != result )
        goto LABEL_5;
      return (_BYTE *)sub_16E7EE0(a2, "\n", 1);
    }
  }
  else
  {
    qmemcpy(v5, " Added Flags: ", 14);
    *(_QWORD *)(v3 + 24) += 14LL;
    v6 = *(_DWORD *)(a1 + 48);
    if ( (v6 & 1) == 0 )
      goto LABEL_3;
  }
  sub_1263B40(a2, "<nusw>");
  if ( (*(_DWORD *)(a1 + 48) & 2) != 0 )
    goto LABEL_8;
LABEL_4:
  result = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) != result )
  {
LABEL_5:
    *result = 10;
    ++*(_QWORD *)(a2 + 24);
    return result;
  }
  return (_BYTE *)sub_16E7EE0(a2, "\n", 1);
}
