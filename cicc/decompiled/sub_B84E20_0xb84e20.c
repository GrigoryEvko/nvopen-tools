// Function: sub_B84E20
// Address: 0xb84e20
//
__int64 __fastcall sub_B84E20(__int64 a1, __int16 a2, __int64 a3, int a4, int a5, __int64 a6, __int64 a7, __int64 a8)
{
  _QWORD *v9; // rdi
  unsigned int v10; // r12d
  __int64 v12; // r14
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  _QWORD *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // r8
  __int64 v21; // rcx
  int v22; // r13d
  _QWORD *v23; // r14
  _QWORD *v24; // rax
  _QWORD *v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rax
  int v29; // edi
  __int64 v30; // r8
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // [rsp-8h] [rbp-50h]
  int v34; // [rsp+14h] [rbp-34h] BYREF
  unsigned __int64 v35; // [rsp+18h] [rbp-30h] BYREF
  int *v36[5]; // [rsp+20h] [rbp-28h] BYREF

  v9 = (_QWORD *)(a1 + 216);
  v34 = 0;
  v10 = sub_C55830((_DWORD)v9, a1, a4, a5, a7, a8, (__int64)&v34);
  if ( (_BYTE)v10 )
    return v10;
  *(_QWORD *)(a1 + 144) -= 4LL;
  *(_QWORD *)(a1 + 200) -= 4LL;
  *(_WORD *)(a1 + 14) = a2;
  v12 = sub_C52410(v9, a1, v33);
  v13 = v12 + 8;
  v15 = sub_C959E0();
  v16 = *(_QWORD **)(v12 + 16);
  if ( v16 )
  {
    v9 = (_QWORD *)(v12 + 8);
    do
    {
      while ( 1 )
      {
        v17 = v16[2];
        v14 = v16[3];
        if ( v15 <= v16[4] )
          break;
        v16 = (_QWORD *)v16[3];
        if ( !v14 )
          goto LABEL_8;
      }
      v9 = v16;
      v16 = (_QWORD *)v16[2];
    }
    while ( v17 );
LABEL_8:
    if ( (_QWORD *)v13 != v9 && v15 >= v9[4] )
      v13 = (__int64)v9;
  }
  if ( v13 == sub_C52410(v9, v15, v14) + 8 || (v19 = *(_QWORD *)(v13 + 56), v20 = (_QWORD *)(v13 + 48), !v19) )
  {
    v22 = -1;
  }
  else
  {
    v15 = *(unsigned int *)(a1 + 8);
    v9 = (_QWORD *)(v13 + 48);
    do
    {
      while ( 1 )
      {
        v21 = *(_QWORD *)(v19 + 16);
        v18 = *(_QWORD *)(v19 + 24);
        if ( *(_DWORD *)(v19 + 32) >= (int)v15 )
          break;
        v19 = *(_QWORD *)(v19 + 24);
        if ( !v18 )
          goto LABEL_17;
      }
      v9 = (_QWORD *)v19;
      v19 = *(_QWORD *)(v19 + 16);
    }
    while ( v21 );
LABEL_17:
    v22 = -1;
    if ( v20 != v9 && (int)v15 >= *((_DWORD *)v9 + 8) )
      v22 = *((_DWORD *)v9 + 9) - 1;
  }
  v23 = (_QWORD *)sub_C52410(v9, v15, v18);
  v35 = sub_C959E0();
  v24 = (_QWORD *)v23[2];
  v25 = v23 + 1;
  if ( !v24 )
    goto LABEL_27;
  do
  {
    while ( 1 )
    {
      v26 = v24[2];
      v27 = v24[3];
      if ( v35 <= v24[4] )
        break;
      v24 = (_QWORD *)v24[3];
      if ( !v27 )
        goto LABEL_25;
    }
    v25 = v24;
    v24 = (_QWORD *)v24[2];
  }
  while ( v26 );
LABEL_25:
  if ( v23 + 1 == v25 || v35 < v25[4] )
  {
LABEL_27:
    v36[0] = (int *)&v35;
    v25 = (_QWORD *)sub_B84CA0(v23, v25, (unsigned __int64 **)v36);
  }
  v28 = v25[7];
  if ( v28 )
  {
    v29 = *(_DWORD *)(a1 + 8);
    v30 = (__int64)(v25 + 6);
    do
    {
      while ( 1 )
      {
        v31 = *(_QWORD *)(v28 + 16);
        v32 = *(_QWORD *)(v28 + 24);
        if ( *(_DWORD *)(v28 + 32) >= v29 )
          break;
        v28 = *(_QWORD *)(v28 + 24);
        if ( !v32 )
          goto LABEL_33;
      }
      v30 = v28;
      v28 = *(_QWORD *)(v28 + 16);
    }
    while ( v31 );
LABEL_33:
    if ( v25 + 6 != (_QWORD *)v30 && v29 >= *(_DWORD *)(v30 + 32) )
      goto LABEL_36;
  }
  else
  {
    v30 = (__int64)(v25 + 6);
  }
  v36[0] = (int *)(a1 + 8);
  v30 = sub_B84D70(v25 + 5, v30, v36);
LABEL_36:
  *(_DWORD *)(v30 + 36) = v22;
  return v10;
}
