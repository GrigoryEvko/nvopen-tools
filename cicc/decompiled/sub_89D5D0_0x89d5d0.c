// Function: sub_89D5D0
// Address: 0x89d5d0
//
__int64 __fastcall sub_89D5D0(unsigned __int64 a1, __int64 a2)
{
  _DWORD *v2; // r12
  unsigned int v3; // r15d
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 result; // rax
  int v7; // r13d
  __int64 v8; // rcx
  unsigned __int16 v9; // ax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r13
  int v17; // ebx
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rax
  _DWORD v25[20]; // [rsp+0h] [rbp-50h] BYREF

  v2 = (_DWORD *)a1;
  v3 = dword_4F06650[0];
  if ( !*(_DWORD *)(a1 + 124) )
  {
    a1 = 1;
    sub_7BDB60(1);
    v2[31] = 1;
  }
  sub_5CCA00();
  result = word_4F06418[0];
  v7 = 0;
  LOBYTE(v8) = word_4F06418[0] == 73 || (word_4F06418[0] & 0xFFF7) == 1;
  if ( (_BYTE)v8 )
    goto LABEL_23;
  while ( 1 )
  {
    v10 = (unsigned int)(result - 27);
    if ( (((_WORD)result - 27) & 0xFFEF) == 0 )
      goto LABEL_23;
    if ( (_WORD)result == 153 )
      break;
    v9 = word_4F06418[0];
LABEL_6:
    if ( v9 != 9 )
      sub_7B8B50(a1, (unsigned int *)a2, v10, v8, v4, v5);
    sub_5CCA00();
    result = word_4F06418[0];
    LOBYTE(v8) = word_4F06418[0] == 73 || (word_4F06418[0] & 0xFFF7) == 1;
    if ( (_BYTE)v8 )
      goto LABEL_23;
  }
  sub_7B8B50(a1, (unsigned int *)a2, v10, v8, v4, v5);
  v9 = word_4F06418[0];
  if ( word_4F06418[0] != 101 && word_4F06418[0] != 151 )
  {
    v7 = 1;
    goto LABEL_6;
  }
  sub_7B8B50(a1, (unsigned int *)a2, v10, v8, v4, v5);
  sub_5CCA00();
  result = word_4F06418[0];
  if ( word_4F06418[0] != 146 )
  {
    v14 = (__int64)&dword_4F077C4;
    if ( dword_4F077C4 != 2 )
    {
      if ( word_4F06418[0] == 1 )
        goto LABEL_17;
      goto LABEL_22;
    }
    if ( word_4F06418[0] != 1 || (word_4D04A10 & 0x200) == 0 )
    {
      a2 = 0;
      a1 = 1025;
      if ( !(unsigned int)sub_7C0F00(0x401u, 0, (__int64)&dword_4F077C4, v11, v12, v13) )
        goto LABEL_22;
    }
LABEL_17:
    a2 = 13;
    a1 = 1025;
    v25[0] = 0;
    v15 = sub_7BF130(0x401u, 13, v25);
    v16 = v15;
    if ( !v15 || (a2 = (__int64)v2, a1 = v15, (v17 = sub_89D460(v15, (__int64)v2)) != 0) )
    {
LABEL_22:
      result = sub_7B8B50(a1, (unsigned int *)a2, v14, v11, v12, v13);
      v7 = 1;
      goto LABEL_23;
    }
    v18 = *(unsigned __int8 *)(v16 + 80);
    if ( (_BYTE)v18 != 3 )
    {
      v14 = (__int64)&qword_4D04A00;
      if ( !xmmword_4D04A20.m128i_i64[1] )
      {
        v14 = (unsigned int)(v18 - 4);
        if ( (unsigned __int8)(v18 - 4) <= 1u )
        {
          if ( (*(_BYTE *)(*(_QWORD *)(v16 + 88) + 177LL) & 0x10) == 0 )
            goto LABEL_22;
          v16 = *(_QWORD *)(*(_QWORD *)(v16 + 96) + 72LL);
          if ( !v16 )
            goto LABEL_22;
LABEL_39:
          LOBYTE(v18) = *(_BYTE *)(v16 + 80);
        }
      }
      if ( (_BYTE)v18 == 19 )
      {
        v23 = *(_QWORD *)(v16 + 88);
        if ( (*(_BYTE *)(v23 + 160) & 2) == 0 )
        {
          a1 = **(_QWORD **)(v23 + 32);
          if ( a1 )
            v17 = *(_DWORD *)(sub_892BC0(a1) + 4);
          v2[27] = 1;
          v2[43] = v17;
        }
      }
      goto LABEL_22;
    }
    if ( !*(_BYTE *)(v16 + 104) )
      goto LABEL_22;
    v24 = *(_QWORD *)(v16 + 88);
    if ( (*(_BYTE *)(v24 + 177) & 0x10) == 0 )
      goto LABEL_22;
    if ( !*(_QWORD *)(*(_QWORD *)(v24 + 168) + 168LL) )
      goto LABEL_22;
    a1 = v16;
    v16 = sub_880FE0(v16);
    if ( !v16 )
      goto LABEL_22;
    goto LABEL_39;
  }
  v2[27] = 1;
  v7 = 1;
  v2[43] = 1;
LABEL_23:
  if ( dword_4F06650[0] != v3 )
  {
    sub_7ADF70((__int64)v25, 0);
    sub_7AE700((__int64)(qword_4F061C0 + 3), v3, dword_4F06650[0], 0, (__int64)v25);
    result = sub_7BC000((unsigned __int64)v25, v3, v19, v20, v21, v22);
  }
  v2[4] = v7;
  return result;
}
