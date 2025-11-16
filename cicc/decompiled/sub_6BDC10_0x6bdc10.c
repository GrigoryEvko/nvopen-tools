// Function: sub_6BDC10
// Address: 0x6bdc10
//
__int64 __fastcall sub_6BDC10(unsigned __int16 a1, int a2, unsigned int a3, int a4)
{
  int v5; // r15d
  _QWORD *v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rax
  int v9; // edx
  unsigned int i; // r15d
  __int64 *v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rcx
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = -(a4 == 0);
  v6 = 0;
  v7 = 0;
  v8 = qword_4D03C50;
  LOWORD(v5) = 0;
  v9 = 0;
  for ( i = v5 + 65537; qword_4D03C50; v8 = qword_4D03C50 )
  {
    v11 = *(__int64 **)(v8 + 136);
    if ( !v11 )
      break;
    v12 = *v11;
    if ( !*v11 )
      break;
    v6 = (_QWORD *)v11[1];
    sub_6E1BE0(v11);
    if ( !a3 )
      sub_832E80(v12);
    v9 = 1;
    v7 = v12;
  }
  if ( word_4F06418[0] != a1 )
  {
    v13 = qword_4F061C8;
    ++*(_BYTE *)(qword_4F061C8 + a1 + 8LL);
    ++*(_QWORD *)(v8 + 40);
    ++*(_BYTE *)(v13 + 75);
    if ( !v9 )
      goto LABEL_13;
    if ( !(unsigned int)sub_7BE800(67) )
      goto LABEL_30;
    while ( 1 )
    {
      if ( a2 && a1 == word_4F06418[0] )
      {
LABEL_30:
        v17 = qword_4F061C8;
        --*(_BYTE *)(qword_4F061C8 + 75LL);
        --*(_BYTE *)(v17 + a1 + 8);
        --*(_QWORD *)(qword_4D03C50 + 40LL);
        return v7;
      }
LABEL_13:
      if ( (unsigned int)sub_869470(v20) )
        break;
      v16 = (__int64)v6;
LABEL_29:
      v6 = (_QWORD *)v16;
      if ( !(unsigned int)sub_7BE800(67) )
        goto LABEL_30;
    }
    while ( word_4F06418[0] != 73 || !dword_4D04428 )
    {
      if ( a3 )
      {
        v16 = sub_6A2C00(1, i);
        goto LABEL_16;
      }
      v16 = sub_6A2B80(i);
      if ( !v7 )
      {
LABEL_26:
        v7 = v16;
        goto LABEL_18;
      }
LABEL_17:
      *v6 = v16;
LABEL_18:
      v14 = sub_867630(v20[0], 0);
      if ( v14 )
      {
        v15 = *(_BYTE *)(v16 + 8) == 0;
        *(_QWORD *)(v16 + 16) = v14;
        if ( v15 )
          *(_QWORD *)(*(_QWORD *)(v16 + 24) + 136LL) = v14;
      }
      if ( !(unsigned int)sub_866C00(v20[0]) )
        goto LABEL_29;
      v6 = (_QWORD *)v16;
    }
    v16 = sub_6BA760(a3, 0);
LABEL_16:
    if ( !v7 )
      goto LABEL_26;
    goto LABEL_17;
  }
  return v7;
}
