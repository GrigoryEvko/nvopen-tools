// Function: sub_150F620
// Address: 0x150f620
//
__int64 __fastcall sub_150F620(__int64 a1, __int64 *a2)
{
  char v2; // al
  unsigned int v3; // r8d
  __int64 v4; // r10
  unsigned __int64 v5; // r9
  unsigned __int64 v6; // rax
  unsigned int v7; // r12d
  unsigned __int64 *v8; // r11
  unsigned int v9; // eax
  __int64 v10; // rbx
  unsigned int v11; // r13d
  unsigned __int64 v12; // r9
  __int64 v13; // rax
  unsigned __int64 v14; // rsi
  __int64 v15; // rdx
  char v16; // cl
  __int64 v17; // rbx
  unsigned int v18; // r9d
  __int64 v19; // r11
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // r8
  unsigned int v22; // r14d
  unsigned __int64 v23; // r10
  _QWORD *v24; // r12
  unsigned int v25; // r8d
  unsigned __int64 v26; // rax
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rsi
  unsigned int v30; // r8d
  __int64 v31; // r10
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdx
  char v35; // cl
  unsigned __int64 v36; // rdx

  v2 = (*((_BYTE *)a2 + 8) >> 1) & 7;
  if ( v2 == 2 )
    return sub_150F320(a1, *a2);
  if ( v2 == 4 )
  {
    v3 = *(_DWORD *)(a1 + 32);
    if ( v3 > 5 )
    {
      v28 = *(_QWORD *)(a1 + 24);
      *(_DWORD *)(a1 + 32) = v3 - 6;
      *(_QWORD *)(a1 + 24) = v28 >> 6;
      LODWORD(v29) = v28 & 0x3F;
      return aAbcdefghijklmn[(unsigned int)v29];
    }
    v4 = 0;
    if ( v3 )
      v4 = *(_QWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    v6 = *(_QWORD *)(a1 + 8);
    v7 = 6 - v3;
    if ( v5 < v6 )
    {
      v8 = (unsigned __int64 *)(v5 + *(_QWORD *)a1);
      if ( v6 >= v5 + 8 )
      {
        v14 = *v8;
        *(_QWORD *)(a1 + 16) = v5 + 8;
        v11 = 64;
LABEL_29:
        *(_QWORD *)(a1 + 24) = v14 >> v7;
        *(_DWORD *)(a1 + 32) = v3 + v11 - 6;
        v29 = v4 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v3 + 58)) & v14) << v3);
        return aAbcdefghijklmn[(unsigned int)v29];
      }
      v9 = v6 - v5;
      *(_QWORD *)(a1 + 24) = 0;
      v10 = v9;
      v11 = 8 * v9;
      v12 = v9 + v5;
      if ( v9 )
      {
        v13 = 0;
        v14 = 0;
        do
        {
          v15 = *((unsigned __int8 *)v8 + v13);
          v16 = 8 * v13++;
          v14 |= v15 << v16;
          *(_QWORD *)(a1 + 24) = v14;
        }
        while ( v10 != v13 );
        *(_QWORD *)(a1 + 16) = v12;
        *(_DWORD *)(a1 + 32) = v11;
        if ( v7 > v11 )
          goto LABEL_12;
        goto LABEL_29;
      }
      *(_QWORD *)(a1 + 16) = v12;
LABEL_32:
      *(_DWORD *)(a1 + 32) = 0;
    }
LABEL_12:
    sub_16BD130("Unexpected end of file", 1);
  }
  v17 = *a2;
  v18 = *(_DWORD *)(a1 + 32);
  if ( (unsigned int)*a2 <= v18 )
  {
    v36 = *(_QWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 32) = v18 - v17;
    *(_QWORD *)(a1 + 24) = v36 >> v17;
    return v36 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17));
  }
  else
  {
    v19 = 0;
    if ( v18 )
      v19 = *(_QWORD *)(a1 + 24);
    v20 = *(_QWORD *)(a1 + 16);
    v21 = *(_QWORD *)(a1 + 8);
    v22 = v17 - v18;
    if ( v20 >= v21 )
      goto LABEL_12;
    v23 = v20 + 8;
    v24 = (_QWORD *)(v20 + *(_QWORD *)a1);
    if ( v21 < v20 + 8 )
    {
      *(_QWORD *)(a1 + 24) = 0;
      v30 = v21 - v20;
      if ( !v30 )
        goto LABEL_32;
      v31 = v30;
      v32 = 0;
      v33 = 0;
      do
      {
        v34 = *((unsigned __int8 *)v24 + v32);
        v35 = 8 * v32++;
        v33 |= v34 << v35;
        *(_QWORD *)(a1 + 24) = v33;
      }
      while ( v30 != v32 );
      v25 = 8 * v30;
      v23 = v20 + v31;
    }
    else
    {
      v25 = 64;
      *(_QWORD *)(a1 + 24) = *v24;
    }
    *(_QWORD *)(a1 + 16) = v23;
    *(_DWORD *)(a1 + 32) = v25;
    if ( v22 > v25 )
      goto LABEL_12;
    v26 = *(_QWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 32) = v18 - v17 + v25;
    *(_QWORD *)(a1 + 24) = v26 >> v22;
    return v19 | ((v26 & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v18 - (unsigned __int8)v17 + 64))) << v18);
  }
}
