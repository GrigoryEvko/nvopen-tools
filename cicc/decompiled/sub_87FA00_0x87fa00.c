// Function: sub_87FA00
// Address: 0x87fa00
//
_QWORD *__fastcall sub_87FA00(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  int v6; // r12d
  _QWORD *result; // rax
  __int64 v8; // r14
  char *v9; // rax

  v5 = qword_4F600D8;
  v6 = a3;
  if ( !qword_4F600D8 )
  {
    v8 = sub_877070(a1, 0, a3, a4);
    qword_4F600D8 = v8;
    v9 = (char *)sub_7279A0(10);
    strcpy(v9, "<unnamed>");
    v5 = qword_4F600D8;
    *(_QWORD *)(v8 + 8) = v9;
    *(_BYTE *)(v8 + 73) |= 1u;
    *(_QWORD *)(v8 + 16) = 9;
  }
  result = sub_87EBB0(a1, v5, a2);
  *((_DWORD *)result + 10) = v6;
  return result;
}
