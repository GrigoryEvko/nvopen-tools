// Function: sub_14ED8F0
// Address: 0x14ed8f0
//
__int64 __fastcall sub_14ED8F0(__int64 a1)
{
  unsigned int v1; // r10d
  unsigned __int64 v2; // r9
  unsigned __int64 v3; // r11
  __int64 v4; // rbx
  unsigned int v5; // r13d
  unsigned __int64 *v6; // r8
  unsigned __int64 v7; // rsi
  unsigned int v8; // r14d
  unsigned int v9; // r8d
  unsigned __int64 v10; // rsi
  __int64 v11; // r13
  unsigned int v12; // r14d
  unsigned __int64 *v13; // r10
  unsigned __int64 v14; // rsi
  unsigned int v15; // r12d
  unsigned __int64 v16; // rax
  char v17; // cl
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 *v20; // r8
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rcx
  unsigned int v23; // r8d
  unsigned __int64 v24; // rdx
  unsigned int v25; // eax
  int v27; // eax
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdx
  char v31; // cl
  unsigned int v32; // r12d
  __int64 v33; // rax
  __int64 v34; // rdx
  char v35; // cl
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rdx
  int v39; // eax
  __int64 v40; // r10
  unsigned int v41; // ebx
  __int64 v42; // rax
  unsigned __int64 v43; // rsi
  __int64 v44; // rdx
  char v45; // cl
  unsigned int v46; // ebx
  unsigned __int64 *v47; // r10
  unsigned int v48; // r11d
  unsigned int v49; // r12d
  unsigned __int64 v50; // rdx
  __int64 v51; // rsi
  unsigned __int64 v52; // r9
  __int64 v53; // r8
  char v54; // cl

  v1 = *(_DWORD *)(a1 + 32);
  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v1 > 3 )
  {
    v36 = *(_QWORD *)(a1 + 24);
    v9 = v1 - 4;
    *(_DWORD *)(a1 + 32) = v1 - 4;
    *(_QWORD *)(a1 + 24) = v36 >> 4;
    LOBYTE(v10) = v36 & 0xF;
  }
  else
  {
    v4 = 0;
    if ( v1 )
      v4 = *(_QWORD *)(a1 + 24);
    v5 = 4 - v1;
    if ( v2 >= v3 )
      goto LABEL_51;
    v6 = (unsigned __int64 *)(v2 + *(_QWORD *)a1);
    if ( v2 + 8 > v3 )
    {
      *(_QWORD *)(a1 + 24) = 0;
      v27 = v3 - v2;
      v28 = (unsigned int)(v3 - v2);
      v8 = 8 * (v3 - v2);
      v2 += v28;
      if ( !v27 )
      {
LABEL_54:
        *(_QWORD *)(a1 + 16) = v2;
        *(_DWORD *)(a1 + 32) = 0;
        goto LABEL_51;
      }
      v29 = 0;
      v7 = 0;
      do
      {
        v30 = *((unsigned __int8 *)v6 + v29);
        v31 = 8 * v29++;
        v7 |= v30 << v31;
        *(_QWORD *)(a1 + 24) = v7;
      }
      while ( v28 != v29 );
      *(_QWORD *)(a1 + 16) = v2;
      *(_DWORD *)(a1 + 32) = v8;
      if ( v5 > v8 )
        goto LABEL_51;
    }
    else
    {
      v7 = *v6;
      *(_QWORD *)(a1 + 16) = v2 + 8;
      v2 += 8LL;
      v8 = 64;
    }
    v9 = v1 + v8 - 4;
    *(_DWORD *)(a1 + 32) = v9;
    *(_QWORD *)(a1 + 24) = v7 >> v5;
    v10 = v4 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v1 + 60)) & v7) << v1);
  }
  if ( (v10 & 8) != 0 )
  {
    do
    {
      while ( v9 > 3 )
      {
        v18 = *(_QWORD *)(a1 + 24);
        v9 -= 4;
        *(_DWORD *)(a1 + 32) = v9;
        *(_QWORD *)(a1 + 24) = v18 >> 4;
        if ( (v18 & 8) == 0 )
          goto LABEL_18;
      }
      LOBYTE(v11) = 0;
      if ( v9 )
        v11 = *(_QWORD *)(a1 + 24);
      v12 = 4 - v9;
      if ( v2 >= v3 )
        goto LABEL_51;
      v13 = (unsigned __int64 *)(v2 + *(_QWORD *)a1);
      if ( v2 + 8 > v3 )
      {
        *(_QWORD *)(a1 + 24) = 0;
        v32 = v3 - v2;
        if ( (_DWORD)v3 == (_DWORD)v2 )
          goto LABEL_54;
        v33 = 0;
        v14 = 0;
        do
        {
          v34 = *((unsigned __int8 *)v13 + v33);
          v35 = 8 * v33++;
          v14 |= v34 << v35;
          *(_QWORD *)(a1 + 24) = v14;
        }
        while ( v32 != v33 );
        v2 += v32;
        v15 = 8 * v32;
        *(_QWORD *)(a1 + 16) = v2;
        *(_DWORD *)(a1 + 32) = v15;
        if ( v12 > v15 )
          goto LABEL_51;
      }
      else
      {
        v14 = *v13;
        *(_QWORD *)(a1 + 16) = v2 + 8;
        v2 += 8LL;
        v15 = 64;
      }
      *(_DWORD *)(a1 + 32) = v9 + v15 - 4;
      *(_QWORD *)(a1 + 24) = v14 >> v12;
      v16 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v9 + 60);
      v17 = v9;
      v9 = v9 + v15 - 4;
    }
    while ( (((unsigned __int8)((v16 & v14) << v17) | (unsigned __int8)v11) & 8) != 0 );
  }
LABEL_18:
  if ( v9 > 0x1F )
  {
    v37 = *(_QWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 32) = 0;
    v38 = v37 >> ((unsigned __int8)v9 - 32);
    *(_QWORD *)(a1 + 24) = HIDWORD(v38);
    v22 = 32LL * (unsigned int)v38 + 8 * v2;
LABEL_37:
    if ( v2 >= v3 )
      return 1;
    goto LABEL_22;
  }
  *(_DWORD *)(a1 + 32) = 0;
  if ( v2 >= v3 )
    goto LABEL_51;
  v19 = v2 + 8;
  v20 = (unsigned __int64 *)(v2 + *(_QWORD *)a1);
  if ( v2 + 8 <= v3 )
  {
    v21 = *v20;
    *(_QWORD *)(a1 + 16) = v19;
    *(_DWORD *)(a1 + 32) = 32;
    *(_QWORD *)(a1 + 24) = HIDWORD(v21);
    v22 = 32LL * (unsigned int)v21 + 8 * v19 - 32;
    goto LABEL_22;
  }
  *(_QWORD *)(a1 + 24) = 0;
  v39 = v3 - v2;
  v40 = (unsigned int)(v3 - v2);
  v41 = 8 * (v3 - v2);
  v2 += v40;
  if ( !v39 )
  {
    *(_QWORD *)(a1 + 16) = v2;
    goto LABEL_51;
  }
  v42 = 0;
  v43 = 0;
  do
  {
    v44 = *((unsigned __int8 *)v20 + v42);
    v45 = 8 * v42++;
    v43 |= v44 << v45;
    *(_QWORD *)(a1 + 24) = v43;
  }
  while ( v40 != v42 );
  *(_QWORD *)(a1 + 16) = v2;
  *(_DWORD *)(a1 + 32) = v41;
  if ( v41 <= 0x1F )
    goto LABEL_51;
  v46 = v41 - 32;
  *(_DWORD *)(a1 + 32) = v46;
  *(_QWORD *)(a1 + 24) = HIDWORD(v43);
  v22 = 32LL * (unsigned int)v43 + 8 * v2 - v46;
  if ( !v46 )
    goto LABEL_37;
LABEL_22:
  v23 = 1;
  if ( v22 >> 3 <= v3 )
  {
    *(_DWORD *)(a1 + 32) = 0;
    v23 = 0;
    *(_QWORD *)(a1 + 16) = (v22 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
    v24 = (v22 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
    v25 = v22 & 0x3F;
    if ( (v22 & 0x3F) != 0 )
    {
      if ( v24 >= v3 )
        goto LABEL_51;
      v47 = (unsigned __int64 *)(v24 + *(_QWORD *)a1);
      if ( v24 + 8 <= v3 )
      {
        v52 = *v47;
        *(_QWORD *)(a1 + 16) = v24 + 8;
        v49 = 64;
      }
      else
      {
        v48 = v3 - v24;
        *(_QWORD *)(a1 + 24) = 0;
        v49 = 8 * v48;
        v50 = v48 + v24;
        if ( !v48 )
        {
          *(_QWORD *)(a1 + 16) = v50;
          goto LABEL_51;
        }
        v51 = 0;
        v52 = 0;
        do
        {
          v53 = *((unsigned __int8 *)v47 + v51);
          v54 = 8 * v51++;
          v52 |= v53 << v54;
          *(_QWORD *)(a1 + 24) = v52;
        }
        while ( v48 != v51 );
        *(_QWORD *)(a1 + 16) = v50;
        *(_DWORD *)(a1 + 32) = v49;
        if ( v25 > v49 )
LABEL_51:
          sub_16BD130("Unexpected end of file", 1);
      }
      v23 = 0;
      *(_DWORD *)(a1 + 32) = v49 - v25;
      *(_QWORD *)(a1 + 24) = v52 >> v25;
    }
  }
  return v23;
}
