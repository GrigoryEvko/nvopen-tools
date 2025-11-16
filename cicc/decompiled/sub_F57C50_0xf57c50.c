// Function: sub_F57C50
// Address: 0xf57c50
//
void __fastcall sub_F57C50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v5; // r15
  __int64 v6; // rsi
  __int64 *v7; // r14
  __int64 v8; // rsi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  int v11; // edx
  unsigned __int64 v12; // rax
  __int64 v14[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = a3 + 48;
  v5 = *(_QWORD *)(a3 + 56);
  if ( a3 + 48 != v5 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v5 )
        {
          sub_B44E20(0);
          BUG();
        }
        sub_B44E20((unsigned __int8 *)(v5 - 24));
        if ( (*(_BYTE *)(v5 - 17) & 8) != 0 )
          sub_F57B90(v5 - 24);
        sub_B44570(v5 - 24);
        if ( !sub_B46AA0(v5 - 24) )
          break;
        v5 = sub_B43D60((_QWORD *)(v5 - 24));
        if ( v3 == v5 )
          goto LABEL_18;
      }
      v6 = *(_QWORD *)(a2 + 48);
      v7 = (__int64 *)(v5 + 24);
      v14[0] = v6;
      if ( v6 )
        break;
      if ( v7 != v14 )
      {
        v8 = *(_QWORD *)(v5 + 24);
        if ( v8 )
          goto LABEL_14;
      }
LABEL_6:
      v5 = *(_QWORD *)(v5 + 8);
      if ( v3 == v5 )
        goto LABEL_18;
    }
    sub_B96E90((__int64)v14, v6, 1);
    if ( v7 == v14 )
    {
      if ( v14[0] )
        sub_B91220((__int64)v14, v14[0]);
      goto LABEL_6;
    }
    v8 = *(_QWORD *)(v5 + 24);
    if ( v8 )
LABEL_14:
      sub_B91220(v5 + 24, v8);
    v9 = (unsigned __int8 *)v14[0];
    *(_QWORD *)(v5 + 24) = v14[0];
    if ( v9 )
      sub_B976B0((__int64)v14, v9, v5 + 24);
    goto LABEL_6;
  }
LABEL_18:
  v10 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == v10 )
  {
    v12 = 0;
  }
  else
  {
    if ( !v10 )
      BUG();
    v11 = *(unsigned __int8 *)(v10 - 24);
    v12 = v10 - 24;
    if ( (unsigned int)(v11 - 30) >= 0xB )
      v12 = 0;
  }
  sub_AA80F0(a1, (unsigned __int64 *)(a2 + 24), 0, a3, *(__int64 **)(a3 + 56), 1, (__int64 *)(v12 + 24), 0);
}
