// Function: sub_2ECCA10
// Address: 0x2ecca10
//
__int64 __fastcall sub_2ECCA10(_QWORD *a1, __int64 a2)
{
  _QWORD *v5; // r14
  _QWORD *v6; // r13
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *v14; // r8
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( (unsigned __int8)sub_BB98D0(a1, *(_QWORD *)a2) )
    return 0;
  v5 = sub_C52410();
  v6 = v5 + 1;
  v7 = sub_C959E0();
  v8 = (_QWORD *)v5[2];
  if ( v8 )
  {
    v9 = v5 + 1;
    do
    {
      while ( 1 )
      {
        v10 = v8[2];
        v11 = v8[3];
        if ( v7 <= v8[4] )
          break;
        v8 = (_QWORD *)v8[3];
        if ( !v11 )
          goto LABEL_8;
      }
      v9 = v8;
      v8 = (_QWORD *)v8[2];
    }
    while ( v10 );
LABEL_8:
    if ( v6 != v9 && v7 >= v9[4] )
      v6 = v9;
  }
  if ( v6 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_46;
  v15 = v6[7];
  v14 = v6 + 6;
  if ( !v15 )
    goto LABEL_46;
  v7 = (unsigned int)dword_5020BA8;
  v16 = v6 + 6;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v15 + 16);
      v12 = *(_QWORD *)(v15 + 24);
      if ( *(_DWORD *)(v15 + 32) >= dword_5020BA8 )
        break;
      v15 = *(_QWORD *)(v15 + 24);
      if ( !v12 )
        goto LABEL_17;
    }
    v16 = (_QWORD *)v15;
    v15 = *(_QWORD *)(v15 + 16);
  }
  while ( v13 );
LABEL_17:
  if ( v14 == v16 || dword_5020BA8 < *((_DWORD *)v16 + 8) || !*((_DWORD *)v16 + 9) )
  {
LABEL_46:
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64, _QWORD *))(**(_QWORD **)(a2 + 16)
                                                                                                 + 304LL))(
           *(_QWORD *)(a2 + 16),
           v7,
           v12,
           v13,
           v14) )
    {
      goto LABEL_21;
    }
    return 0;
  }
  if ( !(_BYTE)qword_5020C28 )
    return 0;
LABEL_21:
  v17 = (__int64 *)a1[1];
  v18 = *v17;
  v19 = v17[1];
  if ( v18 == v19 )
LABEL_42:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_50208AC )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_42;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *, __int64, __int64))(**(_QWORD **)(v18 + 8) + 104LL))(
          *(_QWORD *)(v18 + 8),
          &unk_50208AC,
          v19,
          v13);
  v21 = (__int64 *)a1[1];
  v22 = v20 + 200;
  v23 = *v21;
  v24 = v21[1];
  if ( v23 == v24 )
LABEL_41:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_5027190 )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_41;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_5027190);
  v26 = (__int64 *)a1[1];
  v27 = *(_QWORD *)(v25 + 256);
  v28 = *v26;
  v29 = v26[1];
  if ( v28 == v29 )
LABEL_43:
    BUG();
  while ( *(_UNKNOWN **)v28 != &unk_4F86530 )
  {
    v28 += 16;
    if ( v29 == v28 )
      goto LABEL_43;
  }
  v30 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(
                      *(_QWORD *)(v28 + 8),
                      &unk_4F86530)
                  + 176);
  a1[33] = a1;
  v31[0] = v22;
  v31[1] = v30;
  return sub_2ECC8D0(a1 + 25, a2, v27, v31);
}
