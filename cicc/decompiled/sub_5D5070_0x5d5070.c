// Function: sub_5D5070
// Address: 0x5d5070
//
void __fastcall sub_5D5070(__int64 a1)
{
  __int64 v1; // rbp
  __int64 i; // rax
  __int64 v3; // rax
  __int64 v4; // [rsp-38h] [rbp-38h] BYREF
  _QWORD *v5; // [rsp-30h] [rbp-30h]
  __int64 v6; // [rsp-28h] [rbp-28h]
  __int64 v7; // [rsp-20h] [rbp-20h]
  __int64 v8; // [rsp-8h] [rbp-8h]

  if ( (*(_BYTE *)(a1 + 145) & 0x10) != 0 )
  {
    v8 = v1;
    v6 = a1;
    v5 = (_QWORD *)qword_4CF7CA8;
    if ( qword_4CF7CB0 )
      *(_QWORD *)qword_4CF7CA8 = &v4;
    else
      qword_4CF7CB0 = (__int64)&v4;
    qword_4CF7CA8 = (__int64)&v4;
    v4 = 0;
    v7 = qword_4CF7CA0;
    qword_4CF7CA0 += *(_QWORD *)(a1 + 128);
    for ( i = *(_QWORD *)(a1 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v3 = sub_72FD90(*(_QWORD *)(i + 160), 11);
    sub_5D5070(v3);
    qword_4CF7CA8 = (__int64)v5;
    if ( (__int64 *)qword_4CF7CB0 == &v4 )
      qword_4CF7CB0 = 0;
    else
      *v5 = 0;
    qword_4CF7CA0 = v7;
  }
  else
  {
    sub_5D4E40(*(_BYTE **)(a1 + 8), a1);
  }
}
