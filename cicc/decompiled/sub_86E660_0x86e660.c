// Function: sub_86E660
// Address: 0x86e660
//
__int64 *__fastcall sub_86E660(char a1, _DWORD *a2)
{
  unsigned int *v3; // rsi
  __int64 v4; // r14
  _BYTE *v5; // r13
  __int64 v6; // rbx
  __int64 *result; // rax
  __int64 v8; // [rsp+8h] [rbp-218h] BYREF
  _QWORD v9[66]; // [rsp+10h] [rbp-210h] BYREF

  v3 = *(unsigned int **)(qword_4D03B98 + 176LL * unk_4D03B90 + 160);
  v4 = qword_4D03B98 + 176LL * unk_4D03B90;
  if ( !v3 )
    v3 = &dword_4F063F8;
  v5 = sub_86E480(0x14u, v3);
  if ( *(char *)(v4 + 5) >= 0 )
  {
    *(_BYTE *)(v4 + 5) |= 0x80u;
    *(_QWORD *)(v4 + 136) = &v8;
    v8 = 0;
  }
  if ( !dword_4F04C3C )
    sub_86B430((__int64)v5);
  memset(v9, 0, 0x1D8u);
  v9[19] = v9;
  v9[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v9[22]) |= 1u;
  BYTE6(v9[15]) = (4 * (a1 & 1)) | BYTE6(v9[15]) & 0xFB;
  sub_662DE0((unsigned int *)v9, 0);
  if ( a2 )
    *a2 = dword_4D0488C
       || word_4D04898 && (_DWORD)qword_4F077B4 && qword_4F077A0 > 0x765Bu && sub_729F80(dword_4F063F8)
       || (*(_BYTE *)(qword_4D03B98 + 176LL * unk_4D03B90 + 5) & 4) != 0
       || (v9[16] & 0x400000LL) != 0;
  v6 = qword_4D03B98 + 176LL * unk_4D03B90;
  if ( (*(_BYTE *)(v6 + 4) & 0x10) != 0 && !dword_4F04C3C )
    sub_869D70((__int64)v5, 21);
  result = **(__int64 ***)(v6 + 136);
  if ( result )
  {
    *(_QWORD *)(v6 + 136) = 0;
    *((_QWORD *)v5 + 9) = result;
    while ( *((_BYTE *)result + 8) != 7 || *(_BYTE *)(result[2] + 136) > 2u )
    {
      result = (__int64 *)*result;
      if ( !result )
        goto LABEL_22;
    }
    v5[80] |= 1u;
  }
LABEL_22:
  *(_BYTE *)(v6 + 5) &= ~0x80u;
  return result;
}
