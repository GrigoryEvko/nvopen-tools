// Function: sub_30D6380
// Address: 0x30d6380
//
const char *__fastcall sub_30D6380(__int64 a1)
{
  char v1; // al
  _QWORD *v2; // r13
  _QWORD *v3; // r8
  __int64 v4; // r15
  char v5; // dl
  _QWORD *i; // rcx
  unsigned __int64 v7; // rax
  _QWORD *v8; // rbx
  int v9; // esi
  unsigned __int64 v10; // rax
  bool v11; // si
  __int64 v12; // r12
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // [rsp+8h] [rbp-48h]
  _QWORD *v18; // [rsp+10h] [rbp-40h]
  _QWORD *v19; // [rsp+10h] [rbp-40h]
  char v20; // [rsp+1Fh] [rbp-31h]
  char v21; // [rsp+1Fh] [rbp-31h]

  v1 = sub_B2D610(a1, 53);
  v2 = *(_QWORD **)(a1 + 80);
  v3 = (_QWORD *)(a1 + 72);
  if ( v2 == (_QWORD *)(a1 + 72) )
    return 0;
  v4 = 0x8000000000041LL;
  v5 = v1;
  while ( 1 )
  {
    if ( !v2 )
      BUG();
    i = v2 + 3;
    v7 = v2[3] & 0xFFFFFFFFFFFFFFF8LL;
    if ( v2 + 3 == (_QWORD *)v7 )
      goto LABEL_43;
    if ( !v7 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
LABEL_43:
      BUG();
    if ( *(_BYTE *)(v7 - 24) == 33 )
      return "contains indirect branches";
    if ( (*((_WORD *)v2 - 11) & 0x7FFF) != 0 )
    {
      v19 = v3;
      v21 = v5;
      v15 = sub_ACC4F0((__int64)(v2 - 3));
      v5 = v21;
      v3 = v19;
      v16 = *(_QWORD *)(v15 + 16);
      for ( i = v2 + 3; v16; v16 = *(_QWORD *)(v16 + 8) )
      {
        if ( **(_BYTE **)(v16 + 24) != 40 )
          return "blockaddress used outside of callbr";
      }
    }
    v8 = (_QWORD *)v2[4];
    if ( v8 != i )
    {
      while ( 1 )
      {
        if ( !v8 )
          BUG();
        v9 = *((unsigned __int8 *)v8 - 24);
        v10 = (unsigned int)(v9 - 34);
        if ( (unsigned __int8)(v9 - 34) > 0x33u )
          goto LABEL_22;
        v11 = (_BYTE)v9 == 85;
        if ( !_bittest64(&v4, v10) )
          goto LABEL_22;
        v12 = *(v8 - 7);
        if ( !v12 || *(_BYTE *)v12 || *(_QWORD *)(v12 + 24) != v8[7] )
          break;
        if ( a1 == v12 )
          return "recursive call";
        if ( v5 != 1 && v11 )
          goto LABEL_18;
LABEL_27:
        v14 = *(_DWORD *)(v12 + 36);
        switch ( v14 )
        {
          case 216:
            return "disallowed inlining of @llvm.localescape";
          case 375:
            return "contains VarArgs initialized with va_start";
          case 194:
            return "disallowed inlining of @llvm.icall.branch.funnel";
        }
LABEL_22:
        v8 = (_QWORD *)v8[1];
        if ( v8 == i )
          goto LABEL_23;
      }
      if ( v5 == 1 )
        goto LABEL_22;
      v12 = 0;
      if ( !v11 )
        goto LABEL_22;
LABEL_18:
      v20 = v5;
      v17 = i;
      v18 = v3;
      if ( (unsigned __int8)sub_A73ED0(v8 + 6, 53) || (unsigned __int8)sub_B49560((__int64)(v8 - 3), 53) )
        return "exposes returns-twice attribute";
      v5 = v20;
      v3 = v18;
      i = v17;
      if ( !v12 )
        goto LABEL_22;
      goto LABEL_27;
    }
LABEL_23:
    v2 = (_QWORD *)v2[1];
    if ( v3 == v2 )
      return 0;
  }
}
