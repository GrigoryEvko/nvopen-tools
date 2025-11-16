// Function: sub_86E8F0
// Address: 0x86e8f0
//
__int64 sub_86E8F0()
{
  unsigned int *v0; // rsi
  unsigned __int64 v1; // rdi
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  _BYTE *v5; // r14
  __int64 result; // rax
  __int64 v7; // rax

  if ( word_4F06418[0] == 75 )
    sub_854AB0();
  else
    sub_854B40();
  v0 = *(unsigned int **)(qword_4D03B98 + 176LL * unk_4D03B90 + 160);
  if ( !v0 )
    v0 = &dword_4F063F8;
  v1 = 24;
  v5 = sub_86E480(0x18u, v0);
  result = dword_4F04C3C;
  if ( dword_4F04C3C )
  {
    if ( word_4F06418[0] != 75 )
      goto LABEL_7;
  }
  else
  {
    v0 = (unsigned int *)21;
    v1 = (unsigned __int64)v5;
    result = sub_8699D0((__int64)v5, 21, 0);
    if ( word_4F06418[0] != 75 )
      goto LABEL_7;
  }
  v7 = qword_4F063F0;
  *((_QWORD *)v5 + 1) = qword_4F063F0;
  *(_QWORD *)&dword_4F061D8 = v7;
  result = sub_7B8B50(v1, v0, (__int64)&dword_4F061D8, v2, v3, v4);
LABEL_7:
  if ( (v5[41] & 8) != 0 )
  {
    result = qword_4D03B98 + 176LL * unk_4D03B90;
    *(_QWORD *)(result + 168) = v5;
  }
  return result;
}
