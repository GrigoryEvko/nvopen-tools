// Function: sub_7E64F0
// Address: 0x7e64f0
//
void __fastcall sub_7E64F0(__int64 a1)
{
  __int64 v2; // rbx
  __int64 i; // rbx
  _QWORD *j; // rbx
  _QWORD *v5; // r14
  __int64 v6; // rax
  int v7; // eax
  __int64 *v8; // r13
  unsigned int v9; // r15d
  int v10; // r8d
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdi
  int v15; // [rsp+Ch] [rbp-44h]
  unsigned int v16; // [rsp+Ch] [rbp-44h]
  unsigned int v17; // [rsp+10h] [rbp-40h] BYREF
  int v18; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( *(_BYTE *)(a1 + 28) == 17 )
    qword_4F04C50 = a1;
  v2 = *(_QWORD *)(a1 + 104);
  if ( v2 )
  {
    while ( (unsigned __int8)(*(_BYTE *)(v2 + 140) - 9) > 2u || (unsigned int)sub_736DD0(v2) )
    {
LABEL_5:
      v2 = *(_QWORD *)(v2 + 112);
      if ( !v2 )
        goto LABEL_6;
    }
    sub_7E3EE0(v2);
    v5 = *(_QWORD **)(v2 + 168);
    v6 = v5[24];
    if ( v6 && !*(_BYTE *)(v6 + 177) )
    {
      v16 = sub_7E3BF0(v2, &v17, v19, &v18);
      sub_7E63B0(v2, 0, v16, v17);
      v8 = (__int64 *)v5[28];
      v9 = v17;
      v10 = v16;
      if ( !v8 )
      {
LABEL_27:
        v5 = *(_QWORD **)(v2 + 168);
        goto LABEL_28;
      }
    }
    else
    {
      if ( !v5[28] )
      {
LABEL_28:
        v14 = v5[19];
        if ( v14 && (*(_BYTE *)(v14 + 29) & 0x20) == 0 )
          ((void (*)(void))sub_7E64F0)();
        goto LABEL_5;
      }
      v7 = sub_7E3BF0(v2, &v17, v19, &v18);
      v8 = (__int64 *)v5[28];
      v9 = v17;
      v10 = v7;
      if ( !v8 )
        goto LABEL_25;
    }
    do
    {
      v11 = v8[3];
      if ( !*(_QWORD *)(*(_QWORD *)(v11 + 120) + 176LL) )
      {
        v12 = 0;
        v13 = v8[1];
        if ( *((_BYTE *)v8 + 40) )
        {
          v12 = v8[1];
          v13 = *(_QWORD *)(v13 + 56);
        }
        v15 = v10;
        sub_7E5FC0(v13, v12, v8[2], v11, v10, v9);
        v10 = v15;
      }
      v8 = (__int64 *)*v8;
    }
    while ( v8 );
LABEL_25:
    if ( v10 )
      sub_7FCAA0(v2, v5[25], v5[28]);
    goto LABEL_27;
  }
LABEL_6:
  for ( i = *(_QWORD *)(a1 + 168); i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
      sub_7E64F0(*(_QWORD *)(i + 128));
  }
  for ( j = *(_QWORD **)(a1 + 160); j; j = (_QWORD *)*j )
    sub_7E64F0(j);
  if ( *(_BYTE *)(a1 + 28) == 17 )
    qword_4F04C50 = 0;
}
