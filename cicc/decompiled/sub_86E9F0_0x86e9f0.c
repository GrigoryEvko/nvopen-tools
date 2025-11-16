// Function: sub_86E9F0
// Address: 0x86e9f0
//
__int64 __fastcall sub_86E9F0(int a1)
{
  __int64 v1; // rax
  unsigned int *v2; // rsi
  _BOOL4 v3; // r13d
  _QWORD *v4; // r12
  __int64 v5; // rbx
  char v6; // al
  char v7; // al
  __int64 result; // rax
  __int64 v9; // r14
  __int64 *v10; // [rsp+8h] [rbp-28h] BYREF

  v1 = qword_4D03B98 + 176LL * unk_4D03B90;
  v2 = *(unsigned int **)(v1 + 160);
  v3 = (*(_BYTE *)(v1 + 5) & 2) != 0;
  if ( !v2 )
    v2 = &dword_4F063F8;
  v4 = sub_86E480(0, v2);
  if ( !dword_4F04C3C )
    sub_8699D0((__int64)v4, 21, 0);
  sub_854980(0, (__int64)v4);
  v5 = sub_6B9820(0, a1, v3, &v10, 0);
  if ( v10 )
  {
    sub_7268E0((__int64)v4, 25);
    v4[9] = v10;
  }
  else if ( v3 && (word_4F06418[0] == 74 || (unsigned __int16)sub_7BE840(0, 0) == 74) )
  {
    sub_7268E0((__int64)v4, 25);
  }
  if ( v5 )
  {
    v4[6] = v5;
    v6 = *(_BYTE *)(v5 + 24);
    if ( v6 == 10 )
    {
      v5 = *(_QWORD *)(v5 + 56);
      v6 = *(_BYTE *)(v5 + 24);
    }
    if ( v6 != 1 )
    {
LABEL_25:
      if ( v6 != 8 )
        goto LABEL_18;
      goto LABEL_26;
    }
    while ( *(_BYTE *)(v5 + 56) == 5 && (unsigned int)sub_8D2600(*(_QWORD *)v5) )
    {
      v5 = *(_QWORD *)(v5 + 72);
      v6 = *(_BYTE *)(v5 + 24);
      if ( v6 != 1 )
        goto LABEL_25;
    }
    v7 = *(_BYTE *)(v5 + 24);
    if ( v7 == 8 )
    {
LABEL_26:
      qword_4F5FD78 = 0;
      dword_4F5FD80 = 0;
      goto LABEL_18;
    }
    if ( v7 == 1 && (unsigned __int8)(*(_BYTE *)(v5 + 56) - 105) <= 4u )
    {
      v9 = **(_QWORD **)(v5 + 72);
      if ( (unsigned int)sub_8D2E30(v9) )
        v9 = sub_8D46C0(v9);
      if ( (unsigned int)sub_8D2310(v9) )
      {
        while ( *(_BYTE *)(v9 + 140) == 12 )
          v9 = *(_QWORD *)(v9 + 160);
        if ( (*(_BYTE *)(*(_QWORD *)(v9 + 168) + 20LL) & 1) != 0 )
          *(__int64 *)((char *)&qword_4F5FD78 + 4) = 0x100000000LL;
      }
    }
  }
LABEL_18:
  if ( word_4F06418[0] == 75 )
    *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
  result = *(_QWORD *)&dword_4F061D8;
  v4[1] = *(_QWORD *)&dword_4F061D8;
  return result;
}
