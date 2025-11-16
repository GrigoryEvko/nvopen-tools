// Function: sub_6F5430
// Address: 0x6f5430
//
__int64 __fastcall sub_6F5430(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        int a5,
        int a6,
        unsigned int a7,
        unsigned int a8,
        unsigned int a9,
        unsigned int a10,
        __int64 a11)
{
  __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r12
  __int64 v20; // r9
  __int64 v21; // r8
  __int64 i; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  char v25; // al
  __int64 v27; // r9
  unsigned __int8 v28; // al
  char v29; // cl
  _BOOL8 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rdi
  bool v36; // zf
  __int64 v37; // rax
  __int64 v38; // rax
  char j; // dl
  __int64 v40; // rax
  __int64 v41; // rax
  _QWORD v44[7]; // [rsp+18h] [rbp-38h] BYREF

  v14 = 5;
  v15 = a2;
  v16 = sub_6EAFA0(5u);
  v18 = a4;
  *(_QWORD *)(v16 + 56) = a1;
  v19 = v16;
  if ( a4 )
    *(_BYTE *)(v16 + 49) |= 1u;
  if ( a6 )
    *(_BYTE *)(v16 + 72) |= 1u;
  v20 = a7;
  if ( a7 )
    *(_BYTE *)(v16 + 72) |= 4u;
  v21 = a8;
  if ( a8 )
    *(_BYTE *)(v16 + 72) |= 8u;
  if ( a5 )
  {
    for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    a2 = **(_QWORD **)(i + 168);
    if ( a6 )
      a2 = *(_QWORD *)a2;
    if ( v15 )
    {
      if ( a2 )
      {
        v23 = v15;
        do
        {
          if ( (*(_BYTE *)(a2 + 33) & 1) != 0 )
            break;
          v23 = *(_QWORD *)(v23 + 16);
          a2 = *(_QWORD *)a2;
          if ( !v23 )
            break;
        }
        while ( a2 );
      }
      v14 = a1;
      v18 = sub_6E1DA0(a1, a2);
      v24 = v15;
      if ( v18 )
      {
        do
        {
          v17 = v24;
          v24 = *(_QWORD *)(v24 + 16);
        }
        while ( v24 );
        *(_QWORD *)(v17 + 16) = v18;
      }
    }
    else
    {
      v14 = a1;
      v41 = sub_6E1DA0(a1, a2);
      if ( v41 )
        v15 = v41;
    }
    *(_QWORD *)(v19 + 64) = v15;
  }
  else
  {
    *(_QWORD *)(v16 + 64) = a2;
    if ( !a1 )
      return v19;
  }
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
    *(_BYTE *)(a1 + 193) |= 0x40u;
  v25 = *(_BYTE *)(a1 + 193);
  if ( (v25 & 4) != 0 )
  {
    a10 = 1;
    if ( (v25 & 6) == 0 )
      goto LABEL_26;
  }
  else
  {
    v14 = a9;
    if ( !a9 || (v25 & 6) == 0 )
    {
LABEL_26:
      if ( word_4D04898 )
        goto LABEL_48;
      goto LABEL_27;
    }
  }
  v44[0] = sub_724DC0(v14, a2, v17, v18, v21, v20);
  if ( (unsigned int)sub_6DEB30(v14, a2) && ((v28 = *(_BYTE *)(qword_4D03C50 + 16LL), a10) || v28 > 3u) )
  {
    v29 = *(_BYTE *)(a1 + 193);
  }
  else
  {
    v29 = *(_BYTE *)(a1 + 193);
    if ( (v29 & 4) == 0 )
      goto LABEL_57;
    v28 = *(_BYTE *)(qword_4D03C50 + 16LL);
  }
  v30 = v28 != 0;
  if ( !(unsigned int)sub_71AAF0(v19, v30, a10, (v29 & 4) != 0, a11, v27) )
  {
LABEL_57:
    sub_724E30(v44);
    if ( word_4D04898 && !a10 )
    {
LABEL_48:
      while ( v15 )
      {
        if ( !*(_BYTE *)(v15 + 24) )
          goto LABEL_27;
        v38 = *(_QWORD *)v15;
        for ( j = *(_BYTE *)(*(_QWORD *)v15 + 140LL); j == 12; j = *(_BYTE *)(v38 + 140) )
          v38 = *(_QWORD *)(v38 + 160);
        if ( !j )
          goto LABEL_27;
        v15 = *(_QWORD *)(v15 + 16);
      }
      if ( (unsigned int)sub_6F50A0(a1, 0, 0, 0, a11, v20) )
      {
        sub_7259F0(v19, 2);
        v40 = sub_72C9A0();
        sub_72F900(v19, v40);
      }
    }
    goto LABEL_27;
  }
  if ( a3 )
    *(_QWORD *)(v44[0] + 128LL) = a3;
  v19 = sub_6EAFA0(2u);
  v34 = sub_724E50(v44, v30, v31, v32, v33);
  sub_72F900(v19, v34);
  v35 = *(_QWORD *)(v19 + 56);
  v36 = *(_BYTE *)(v35 + 173) == 10;
  v44[0] = v35;
  if ( v36 )
  {
    v37 = *(_QWORD *)(v35 + 176);
    if ( v37 )
    {
      if ( (*(_BYTE *)(v35 - 8) & 1) == 0 && (*(_BYTE *)(v37 - 8) & 1) != 0 )
        sub_740190(v35, v35, 0);
    }
  }
LABEL_27:
  if ( (*(_BYTE *)(a1 + 196) & 0x20) != 0 )
    *(_BYTE *)(qword_4D03C50 + 20LL) |= 0x40u;
  return v19;
}
