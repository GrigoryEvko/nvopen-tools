// Function: sub_875110
// Address: 0x875110
//
void sub_875110()
{
  __int64 v0; // rbx
  __int64 v1; // rax
  _BYTE *v2; // r12
  unsigned int v3; // edi
  _QWORD v4[5]; // [rsp+8h] [rbp-28h] BYREF

  dword_4F5FD80 = 0;
  qword_4F5FD68 = 0;
  qword_4F5FD70 = 0;
  v0 = *(_QWORD *)(qword_4F04C50 + 32LL);
  v4[0] = *(_QWORD *)&dword_4F063F8;
  qword_4F5FD78 = 0x100000001LL;
  unk_4D03B90 = -1;
  v1 = sub_86B2C0(0);
  *(_QWORD *)(v1 + 56) = qword_4F06BC0;
  sub_86CBE0(v1);
  v2 = sub_726B30(19);
  **((_BYTE **)v2 + 9) = 1;
  *(_QWORD *)v2 = *(_QWORD *)&dword_4F063F8;
  if ( !dword_4F04C3C )
    sub_8699D0((__int64)v2, 21, 0);
  sub_86E330((__int64)v2);
  if ( (*(_BYTE *)(v0 + 193) & 2) == 0 || dword_4D04884 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(v0 + 174) - 1) > 1u )
      return;
LABEL_10:
    sub_733780(0, 0, 0, 1, 0);
    return;
  }
  v3 = 2397;
  if ( *(_BYTE *)(v0 + 174) != 1 )
    v3 = 2848;
  sub_6851C0(v3, v4);
  if ( (unsigned __int8)(*(_BYTE *)(v0 + 174) - 1) <= 1u )
    goto LABEL_10;
}
