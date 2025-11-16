// Function: sub_34D0480
// Address: 0x34d0480
//
void __fastcall sub_34D0480(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v7; // r14
  _QWORD *v8; // r13
  _QWORD *v9; // r15
  unsigned __int64 v10; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rdx
  int v19; // r8d
  __int64 v20; // rbx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // r14
  __int64 v24; // r15
  char v25; // al
  __int64 v26; // rsi
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  int v29; // [rsp+1Ch] [rbp-44h]
  __int64 v31[7]; // [rsp+28h] [rbp-38h] BYREF

  v31[0] = a2;
  v7 = *(_QWORD *)(a1 + 16);
  v8 = sub_C52410();
  v9 = v8 + 1;
  v10 = sub_C959E0();
  v11 = (_QWORD *)v8[2];
  if ( v11 )
  {
    v12 = v8 + 1;
    do
    {
      while ( 1 )
      {
        v13 = v11[2];
        v14 = v11[3];
        if ( v10 <= v11[4] )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v14 )
          goto LABEL_6;
      }
      v12 = v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v13 );
LABEL_6:
    if ( v9 != v12 && v10 >= v12[4] )
      v9 = v12;
  }
  if ( v9 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_33;
  v15 = v9[7];
  if ( !v15 )
    goto LABEL_33;
  v16 = v9 + 6;
  do
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(v15 + 16);
      v18 = *(_QWORD *)(v15 + 24);
      if ( *(_DWORD *)(v15 + 32) >= (signed int)dword_503A828 )
        break;
      v15 = *(_QWORD *)(v15 + 24);
      if ( !v18 )
        goto LABEL_15;
    }
    v16 = (_QWORD *)v15;
    v15 = *(_QWORD *)(v15 + 16);
  }
  while ( v17 );
LABEL_15:
  if ( v9 + 6 == v16 || (signed int)dword_503A828 < *((_DWORD *)v16 + 8) || *((int *)v16 + 9) <= 0 )
  {
LABEL_33:
    v19 = *(_DWORD *)(*(_QWORD *)(v7 + 200) + 8LL);
    if ( !v19 )
      return;
  }
  else
  {
    v19 = qword_503A868[8];
  }
  v20 = a1 + 8;
  v21 = *(_QWORD *)(v31[0] + 40);
  v22 = *(_QWORD *)(v31[0] + 32);
  if ( v21 == v22 )
  {
LABEL_26:
    *(_BYTE *)(a4 + 49) = 1;
    *(_DWORD *)(a4 + 12) = v19;
    *(_DWORD *)(a4 + 8) = 0;
    *(_DWORD *)(a4 + 16) = 0;
    *(_DWORD *)(a4 + 40) = 2;
    *(_WORD *)(a4 + 44) = 257;
    return;
  }
  while ( 1 )
  {
    v23 = *(_QWORD *)(*(_QWORD *)v22 + 56LL);
    v24 = *(_QWORD *)v22 + 48LL;
    if ( v24 != v23 )
      break;
LABEL_25:
    v22 += 8;
    if ( v21 == v22 )
      goto LABEL_26;
  }
  while ( 1 )
  {
    if ( !v23 )
      BUG();
    v25 = *(_BYTE *)(v23 - 24);
    if ( v25 != 34 && v25 != 85 )
      goto LABEL_24;
    v26 = *(_QWORD *)(v23 - 56);
    if ( !v26 )
      break;
    if ( *(_BYTE *)v26 )
      break;
    if ( *(_QWORD *)(v26 + 24) != *(_QWORD *)(v23 + 56) )
      break;
    v27 = v22;
    v28 = v21;
    v29 = v19;
    if ( (unsigned __int8)sub_DF7D80(v20, (_BYTE *)v26) )
      break;
    v19 = v29;
    v21 = v28;
    v22 = v27;
LABEL_24:
    v23 = *(_QWORD *)(v23 + 8);
    if ( v24 == v23 )
      goto LABEL_25;
  }
  if ( a5 )
    sub_34CF850(a5, v31, (unsigned __int8 *)(v23 - 24));
}
