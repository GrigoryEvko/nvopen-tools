// Function: sub_6BE930
// Address: 0x6be930
//
__int64 __fastcall sub_6BE930(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4, unsigned int *a5)
{
  unsigned int v7; // eax
  bool v8; // bl
  __int64 result; // rax
  unsigned __int64 v10; // rdx
  __int64 i; // rax
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // [rsp+8h] [rbp-48h]
  unsigned __int16 v16; // [rsp+Eh] [rbp-42h]
  int v17; // [rsp+10h] [rbp-40h]
  unsigned int v18; // [rsp+14h] [rbp-3Ch]
  __int64 v20; // [rsp+18h] [rbp-38h]

  v17 = sub_8D3BB0(a1);
  v18 = sub_8DD3B0(a1);
  v15 = dword_4F063F8;
  v16 = word_4F063FC[0];
  v7 = dword_4D0478C;
  v8 = (*(_BYTE *)(qword_4D03C50 + 21LL) & 4) != 0;
  if ( dword_4D0478C )
    v7 = v17 != 0 && v18 == 0;
  *a5 = v7;
  result = sub_8D3A70(a1);
  if ( !(_DWORD)result )
  {
    v12 = 0;
    goto LABEL_21;
  }
  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v12 = *(_QWORD *)(*(_QWORD *)i + 96LL);
  if ( !*(_QWORD *)(v12 + 8) )
  {
    if ( a3 )
    {
      result = v18;
      if ( v18 )
        goto LABEL_17;
      goto LABEL_26;
    }
LABEL_9:
    v20 = v12;
    if ( a2 )
    {
LABEL_10:
      result = sub_690A60(*(_QWORD *)(a2 + 16), a2, (_QWORD *)v10);
      v12 = v20;
      *a4 = result;
LABEL_11:
      if ( v18 || !v12 )
        goto LABEL_17;
      goto LABEL_13;
    }
LABEL_29:
    result = sub_6BDC10(0x1Cu, 0, 1u, 0);
    v12 = v20;
    *a4 = result;
    goto LABEL_11;
  }
  result = v18;
  if ( v18 )
  {
LABEL_21:
    v10 = a3;
    if ( a3 )
      goto LABEL_17;
    v20 = v12;
    if ( a2 )
      goto LABEL_10;
    goto LABEL_29;
  }
  *(_BYTE *)(qword_4D03C50 + 21LL) |= 4u;
  result = a3;
  if ( !a3 )
    goto LABEL_9;
LABEL_13:
  if ( *(_QWORD *)(v12 + 8) )
  {
    *a5 = 0;
    if ( dword_4D0478C && v17 && !(unsigned int)sub_84AA50(*(_QWORD *)(v12 + 8), 2, 0, 0, *a4, 0, 0) )
      *a5 = 1;
    result = 4 * (unsigned int)v8;
    *(_BYTE *)(qword_4D03C50 + 21LL) = (4 * v8) | *(_BYTE *)(qword_4D03C50 + 21LL) & 0xFB;
    goto LABEL_17;
  }
LABEL_26:
  if ( !*a5 )
    return result;
  result = sub_68A710(a1, (_QWORD *)*a4);
  if ( (_DWORD)result )
  {
    *a5 = 0;
    return result;
  }
LABEL_17:
  if ( *a5 )
  {
    v13 = sub_6E2F40(1);
    v14 = *a4;
    *(_DWORD *)(v13 + 32) = v15;
    *(_QWORD *)(v13 + 24) = v14;
    *(_WORD *)(v13 + 36) = v16;
    *(_QWORD *)(v13 + 40) = *(_QWORD *)&dword_4F063F8;
    *a4 = v13;
    result = qword_4D03C50;
    *(_BYTE *)(qword_4D03C50 + 21LL) |= 0x10u;
  }
  return result;
}
