// Function: sub_216E840
// Address: 0x216e840
//
__int64 __fastcall sub_216E840(__int64 a1, __int64 a2, __int64 a3, int *a4)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rax
  _DWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  _DWORD *v13; // r8
  _DWORD *v14; // rdi
  int v15; // esi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // rbx
  __int64 v21; // r13
  char v22; // al
  __int64 v23; // rdx
  unsigned int v24; // eax
  __int64 result; // rax
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  _BYTE **v28; // rax

  v6 = *(_QWORD *)(a1 + 8);
  v7 = sub_16D5D50();
  v8 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v9 = dword_4FA0208;
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
          goto LABEL_6;
      }
      v9 = v8;
      v8 = (_QWORD *)v8[2];
    }
    while ( v10 );
LABEL_6:
    if ( v9 != dword_4FA0208 && v7 >= *((_QWORD *)v9 + 4) )
    {
      v12 = *((_QWORD *)v9 + 7);
      v13 = v9 + 12;
      if ( v12 )
      {
        v14 = v9 + 12;
        v15 = qword_5057460[1];
        do
        {
          while ( 1 )
          {
            v16 = *(_QWORD *)(v12 + 16);
            v17 = *(_QWORD *)(v12 + 24);
            if ( *(_DWORD *)(v12 + 32) >= v15 )
              break;
            v12 = *(_QWORD *)(v12 + 24);
            if ( !v17 )
              goto LABEL_13;
          }
          v14 = (_DWORD *)v12;
          v12 = *(_QWORD *)(v12 + 16);
        }
        while ( v16 );
LABEL_13:
        if ( v13 != v14 && v15 >= v14[8] && (int)v14[9] > 0 )
          goto LABEL_16;
      }
    }
  }
  if ( *(_DWORD *)(*(_QWORD *)(v6 + 160) + 8LL) )
  {
LABEL_16:
    v18 = *(_QWORD *)(a2 + 32);
    v19 = *(_QWORD *)(a2 + 40);
    if ( v18 == v19 )
    {
LABEL_23:
      *((_BYTE *)a4 + 49) = 1;
      a4[2] = 0;
      a4[4] = 0;
      a4[10] = 2;
      goto LABEL_24;
    }
    while ( 1 )
    {
      v20 = *(_QWORD *)(*(_QWORD *)v18 + 48LL);
      v21 = *(_QWORD *)v18 + 40LL;
      if ( v20 != v21 )
        break;
LABEL_22:
      v18 += 8;
      if ( v19 == v18 )
        goto LABEL_23;
    }
    while ( 1 )
    {
      if ( !v20 )
        BUG();
      v22 = *(_BYTE *)(v20 - 8);
      v23 = v20 - 24;
      if ( v22 == 78 )
      {
        v26 = v23 | 4;
      }
      else
      {
        if ( v22 != 29 )
          goto LABEL_21;
        v26 = v23 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v27 = v26 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v26 & 4) != 0 )
        v28 = (_BYTE **)(v27 - 24);
      else
        v28 = (_BYTE **)(v27 - 72);
      if ( (*v28)[16] || sub_216E380(*v28) )
        break;
LABEL_21:
      v20 = *(_QWORD *)(v20 + 8);
      if ( v21 == v20 )
        goto LABEL_22;
    }
  }
LABEL_24:
  v24 = *a4;
  *((_WORD *)a4 + 22) = 257;
  result = v24 >> 1;
  a4[3] = result;
  return result;
}
