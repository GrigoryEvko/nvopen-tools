// Function: sub_87F680
// Address: 0x87f680
//
_QWORD *__fastcall sub_87F680(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  _QWORD *result; // rax
  __int64 v7; // r12
  char *v8; // rax
  _QWORD *v9; // [rsp+8h] [rbp-18h]

  v5 = qword_4F600E0;
  if ( !qword_4F600E0 )
  {
    v9 = a2;
    v7 = sub_877070(a1, 0, a2, a4);
    qword_4F600E0 = v7;
    v8 = (char *)sub_7279A0(10);
    a2 = v9;
    strcpy(v8, "<unnamed>");
    v5 = qword_4F600E0;
    *(_QWORD *)(v7 + 8) = v8;
    *(_BYTE *)(v7 + 73) |= 1u;
    *(_QWORD *)(v7 + 16) = 9;
  }
  result = sub_87EBB0(a1, v5, a2);
  *((_DWORD *)result + 10) = *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C);
  return result;
}
