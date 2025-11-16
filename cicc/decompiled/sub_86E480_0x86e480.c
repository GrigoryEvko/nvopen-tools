// Function: sub_86E480
// Address: 0x86e480
//
_BYTE *__fastcall sub_86E480(unsigned __int8 a1, unsigned int *a2)
{
  unsigned int *v2; // r12
  unsigned __int64 v3; // r15
  _BYTE *v4; // r14
  __int64 v5; // rax
  __int64 v7; // rdx
  __int64 i; // rax
  void *v9; // rdx

  v2 = a2;
  if ( a1 == 7 )
  {
    dword_4F5FD80 = 0;
    v3 = 7;
    qword_4F5FD78 = 0x100000001LL;
    v4 = sub_726B30(7);
    *(_QWORD *)v4 = *(_QWORD *)a2;
    sub_86B010((__int64)v4, qword_4F5FD78);
LABEL_19:
    v9 = &loc_1320000;
    v5 = 176LL * unk_4D03B90;
    if ( _bittest64((const __int64 *)&v9, v3) )
      goto LABEL_8;
    goto LABEL_7;
  }
  v3 = a1;
  if ( (a1 & 0xFD) == 8 )
  {
    v7 = *(_QWORD *)(qword_4F04C50 + 32LL);
    for ( i = *(_QWORD *)(v7 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( (*(_BYTE *)(*(_QWORD *)(i + 168) + 20LL) & 1) != 0 && HIDWORD(qword_4F5FD78) && (*(_BYTE *)(v7 + 195) & 8) == 0 )
    {
      if ( !*a2 )
        a2 = &dword_4F063F8;
      sub_684B30(0x519u, a2);
    }
    v4 = sub_726B30(a1);
    *(_QWORD *)v4 = *(_QWORD *)v2;
    sub_86B010((__int64)v4, qword_4F5FD78);
  }
  else
  {
    v4 = sub_726B30(a1);
    *(_QWORD *)v4 = *(_QWORD *)a2;
    sub_86B010((__int64)v4, qword_4F5FD78);
    if ( a1 != 6 && a1 != 23 )
      goto LABEL_5;
  }
  qword_4F5FD78 = 0;
  dword_4F5FD80 = 0;
LABEL_5:
  if ( a1 <= 0x18u )
    goto LABEL_19;
  v5 = 176LL * unk_4D03B90;
LABEL_7:
  *(_BYTE *)(qword_4D03B98 + v5 + 4) |= 0x80u;
LABEL_8:
  *(_QWORD *)(qword_4D03B98 + v5 + 160) = 0;
  return v4;
}
