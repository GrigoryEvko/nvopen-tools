// Function: sub_7B22D0
// Address: 0x7b22d0
//
__int64 sub_7B22D0()
{
  _QWORD *v0; // rax
  const char *v1; // r13
  char v2; // r12
  size_t v3; // rax
  char *v4; // rax
  char *v5; // rdi
  __int64 result; // rax
  _DWORD v7[9]; // [rsp+Ch] [rbp-24h] BYREF

  v0 = qword_4F04D90;
  if ( qword_4F04D90 )
    goto LABEL_2;
  v0 = (_QWORD *)unk_4F076D8;
  if ( dword_4F04D88[0] )
  {
    qword_4F04D90 = (_QWORD *)unk_4F076D8;
    dword_4F04D88[0] = 0;
    if ( unk_4F076D8 )
    {
LABEL_2:
      v1 = (const char *)v0[1];
      if ( dword_4F07680[0] && sub_722B80(qword_4F076E8, (unsigned __int8 *)qword_4F076B0, 0) )
        sub_720AA0((__int64)qword_4F076B0);
      v2 = dword_4F04D88[0];
      v3 = strlen(v1);
      v4 = (char *)sub_7279A0(v3 + 1);
      v5 = strcpy(v4, v1);
      sub_7B2160(v5, 1, 1u, 0, 1, v2, 0, 0, 0, v7);
      result = *qword_4F04D90;
      qword_4F04D90 = (_QWORD *)*qword_4F04D90;
      if ( v7[0] )
        return sub_7B22D0();
      return result;
    }
    goto LABEL_10;
  }
  if ( !unk_4F076D8 )
  {
LABEL_10:
    result = (__int64)&qword_4F076D0;
    if ( !qword_4F076D0 )
      return result;
  }
  result = dword_4F07680[0];
  if ( dword_4F07680[0] )
    return sub_720AA0((__int64)qword_4F076E8);
  return result;
}
