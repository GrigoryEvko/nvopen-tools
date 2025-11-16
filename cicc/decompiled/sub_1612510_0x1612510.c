// Function: sub_1612510
// Address: 0x1612510
//
__int64 __fastcall sub_1612510(__int64 a1, int a2, __int64 a3, int a4, int a5, __int64 a6, __int64 a7, __int64 a8)
{
  _DWORD *v9; // rdi
  __int64 result; // rax
  unsigned int v11; // r13d
  __int64 v12; // rdx
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  int v16; // r12d
  __int64 v17; // rax
  _DWORD *v18; // r8
  __int64 v19; // rcx
  __int64 v20; // rax
  _QWORD *v21; // rsi
  unsigned __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // rax
  int v27; // edi
  __int64 v28; // r8
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // [rsp-8h] [rbp-50h]
  int v32; // [rsp+14h] [rbp-34h] BYREF
  __int64 v33; // [rsp+18h] [rbp-30h] BYREF
  int *v34[5]; // [rsp+20h] [rbp-28h] BYREF

  v9 = (_DWORD *)(a1 + 208);
  v32 = 0;
  LODWORD(result) = sub_16B3550((_DWORD)v9, a1, a4, a5, a7, a8, (__int64)&v32);
  v11 = result;
  if ( (_BYTE)result )
    return (unsigned int)result;
  *(_QWORD *)(a1 + 168) -= 4LL;
  *(_QWORD *)(a1 + 192) -= 4LL;
  *(_DWORD *)(a1 + 16) = a2;
  v13 = sub_16D5D50(v9, a1, v31);
  v14 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v9 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v15 = v14[2];
        v12 = v14[3];
        if ( v13 <= v14[4] )
          break;
        v14 = (_QWORD *)v14[3];
        if ( !v12 )
          goto LABEL_8;
      }
      v9 = v14;
      v14 = (_QWORD *)v14[2];
    }
    while ( v15 );
LABEL_8:
    v16 = -1;
    if ( v9 != dword_4FA0208 && v13 >= *((_QWORD *)v9 + 4) )
    {
      v17 = *((_QWORD *)v9 + 7);
      v18 = v9 + 12;
      if ( v17 )
      {
        v13 = *(unsigned int *)(a1 + 8);
        v9 += 12;
        do
        {
          while ( 1 )
          {
            v19 = *(_QWORD *)(v17 + 16);
            v12 = *(_QWORD *)(v17 + 24);
            if ( *(_DWORD *)(v17 + 32) >= (int)v13 )
              break;
            v17 = *(_QWORD *)(v17 + 24);
            if ( !v12 )
              goto LABEL_15;
          }
          v9 = (_DWORD *)v17;
          v17 = *(_QWORD *)(v17 + 16);
        }
        while ( v19 );
LABEL_15:
        v16 = -1;
        if ( v18 != v9 && (int)v13 >= v9[8] )
          v16 = v9[9] - 1;
      }
    }
  }
  else
  {
    v16 = -1;
  }
  v20 = sub_16D5D50(v9, v13, v12);
  v21 = dword_4FA0208;
  v33 = v20;
  v22 = v20;
  v23 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_25;
  do
  {
    while ( 1 )
    {
      v24 = v23[2];
      v25 = v23[3];
      if ( v22 <= v23[4] )
        break;
      v23 = (_QWORD *)v23[3];
      if ( !v25 )
        goto LABEL_23;
    }
    v21 = v23;
    v23 = (_QWORD *)v23[2];
  }
  while ( v24 );
LABEL_23:
  if ( v21 == (_QWORD *)dword_4FA0208 || v22 < v21[4] )
  {
LABEL_25:
    v34[0] = (int *)&v33;
    v21 = (_QWORD *)sub_1612390(&qword_4FA0200, v21, (unsigned __int64 **)v34);
  }
  v26 = v21[7];
  if ( v26 )
  {
    v27 = *(_DWORD *)(a1 + 8);
    v28 = (__int64)(v21 + 6);
    do
    {
      while ( 1 )
      {
        v29 = *(_QWORD *)(v26 + 16);
        v30 = *(_QWORD *)(v26 + 24);
        if ( *(_DWORD *)(v26 + 32) >= v27 )
          break;
        v26 = *(_QWORD *)(v26 + 24);
        if ( !v30 )
          goto LABEL_31;
      }
      v28 = v26;
      v26 = *(_QWORD *)(v26 + 16);
    }
    while ( v29 );
LABEL_31:
    if ( v21 + 6 != (_QWORD *)v28 && v27 >= *(_DWORD *)(v28 + 32) )
      goto LABEL_34;
  }
  else
  {
    v28 = (__int64)(v21 + 6);
  }
  v34[0] = (int *)(a1 + 8);
  v28 = sub_1612460(v21 + 5, v28, v34);
LABEL_34:
  *(_DWORD *)(v28 + 36) = v16;
  return v11;
}
