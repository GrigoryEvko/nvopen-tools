// Function: sub_87F730
// Address: 0x87f730
//
_QWORD *__fastcall sub_87F730(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rdx
  __int64 v5; // rsi
  _QWORD *result; // rax
  __int64 v7; // rbx
  char *v8; // rax

  v4 = a1;
  v5 = qword_4F600C0;
  if ( !qword_4F600C0 )
  {
    v7 = sub_877070(a1, 0, a1, a4);
    qword_4F600C0 = v7;
    v8 = (char *)sub_7279A0(10);
    v4 = a1;
    strcpy(v8, "<unnamed>");
    v5 = qword_4F600C0;
    *(_QWORD *)(v7 + 8) = v8;
    *(_BYTE *)(v7 + 73) |= 1u;
    *(_QWORD *)(v7 + 16) = 9;
  }
  result = sub_87EBB0(0x17u, v5, v4);
  *((_DWORD *)result + 10) = *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  return result;
}
