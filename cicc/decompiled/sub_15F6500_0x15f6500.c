// Function: sub_15F6500
// Address: 0x15f6500
//
__int64 __fastcall sub_15F6500(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 v10; // r12
  __int64 *v11; // rbx
  bool v12; // zf
  __int64 v14; // rax
  __int64 v15; // rdi
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // r8
  unsigned __int64 v29; // rdi
  __int64 v30; // rdi
  int v31; // r13d
  __int64 v32; // r14
  _QWORD *v33; // r13
  __int64 *v34; // rcx
  __int64 v35; // r11
  _QWORD *v36; // rax
  __int64 v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // r8
  unsigned __int64 v40; // rdi
  __int64 v41; // rdi
  __int64 v42; // r15
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  int v46; // r12d
  __int64 v47; // r14
  __int64 v48; // rdx
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v52; // [rsp+8h] [rbp-48h]
  __int64 v54; // [rsp+18h] [rbp-38h]

  v10 = a1;
  v11 = a9;
  v12 = *(_QWORD *)(a1 - 72) == 0;
  *(_QWORD *)(a1 + 64) = a2;
  if ( !v12 )
  {
    v14 = *(_QWORD *)(a1 - 56);
    v15 = *(_QWORD *)(a1 - 64);
    v16 = v14 & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v16 = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
  }
  *(_QWORD *)(v10 - 72) = a3;
  if ( a3 )
  {
    v17 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(v10 - 64) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = (v10 - 64) | *(_QWORD *)(v17 + 16) & 3LL;
    *(_QWORD *)(v10 - 56) = (a3 + 8) | *(_QWORD *)(v10 - 56) & 3LL;
    *(_QWORD *)(a3 + 8) = v10 - 72;
  }
  if ( *(_QWORD *)(v10 - 48) )
  {
    v18 = *(_QWORD *)(v10 - 40);
    v19 = *(_QWORD *)(v10 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v19 = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
  }
  *(_QWORD *)(v10 - 48) = a4;
  if ( a4 )
  {
    v20 = *(_QWORD *)(a4 + 8);
    *(_QWORD *)(v10 - 40) = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = (v10 - 40) | *(_QWORD *)(v20 + 16) & 3LL;
    *(_QWORD *)(v10 - 32) = (a4 + 8) | *(_QWORD *)(v10 - 32) & 3LL;
    *(_QWORD *)(a4 + 8) = v10 - 48;
  }
  if ( *(_QWORD *)(v10 - 24) )
  {
    v21 = *(_QWORD *)(v10 - 16);
    v22 = *(_QWORD *)(v10 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v22 = v21;
    if ( v21 )
      *(_QWORD *)(v21 + 16) = *(_QWORD *)(v21 + 16) & 3LL | v22;
  }
  *(_QWORD *)(v10 - 24) = a5;
  if ( a5 )
  {
    v23 = *(_QWORD *)(a5 + 8);
    *(_QWORD *)(v10 - 16) = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 16) = (v10 - 16) | *(_QWORD *)(v23 + 16) & 3LL;
    *(_QWORD *)(v10 - 8) = (a5 + 8) | *(_QWORD *)(v10 - 8) & 3LL;
    *(_QWORD *)(a5 + 8) = v10 - 24;
  }
  v24 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
  v25 = (_QWORD *)(v10 - 24 * v24);
  v26 = (8 * a8) >> 3;
  if ( 8 * a8 > 0 )
  {
    do
    {
      v27 = *a7;
      if ( *v25 )
      {
        v28 = v25[1];
        v29 = v25[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v29 = v28;
        if ( v28 )
          *(_QWORD *)(v28 + 16) = *(_QWORD *)(v28 + 16) & 3LL | v29;
      }
      *v25 = v27;
      if ( v27 )
      {
        v30 = *(_QWORD *)(v27 + 8);
        v25[1] = v30;
        if ( v30 )
          *(_QWORD *)(v30 + 16) = (unsigned __int64)(v25 + 1) | *(_QWORD *)(v30 + 16) & 3LL;
        v25[2] = (v27 + 8) | v25[2] & 3LL;
        *(_QWORD *)(v27 + 8) = v25;
      }
      ++a7;
      v25 += 3;
      --v26;
    }
    while ( v26 );
    v24 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
  }
  v31 = a8;
  v32 = (__int64)a9;
  if ( &a9[7 * a10] != a9 )
  {
    v33 = (_QWORD *)(v10 + 24 * ((unsigned int)a8 - v24));
    do
    {
      v34 = *(__int64 **)(v32 + 32);
      v35 = *(_QWORD *)(v32 + 40) - (_QWORD)v34;
      if ( v35 > 0 )
      {
        v36 = v33;
        v37 = (__int64)(*(_QWORD *)(v32 + 40) - (_QWORD)v34) >> 3;
        do
        {
          v38 = *v34;
          if ( *v36 )
          {
            v39 = v36[1];
            v40 = v36[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v40 = v39;
            if ( v39 )
              *(_QWORD *)(v39 + 16) = *(_QWORD *)(v39 + 16) & 3LL | v40;
          }
          *v36 = v38;
          if ( v38 )
          {
            v41 = *(_QWORD *)(v38 + 8);
            v36[1] = v41;
            if ( v41 )
              *(_QWORD *)(v41 + 16) = (unsigned __int64)(v36 + 1) | *(_QWORD *)(v41 + 16) & 3LL;
            v36[2] = (v38 + 8) | v36[2] & 3LL;
            *(_QWORD *)(v38 + 8) = v36;
          }
          ++v34;
          v36 += 3;
          --v37;
        }
        while ( v37 );
        v33 += 3 * (v35 >> 3);
      }
      v32 += 56;
    }
    while ( &a9[7 * a10] != (__int64 *)v32 );
    v31 = a8;
    v11 = a9;
  }
  v42 = *(_QWORD *)sub_16498A0(v10);
  if ( *(char *)(v10 + 23) < 0 )
  {
    v43 = sub_1648A40(v10);
    v54 = v44 + v43;
    if ( *(char *)(v10 + 23) >= 0 )
      v45 = 0;
    else
      v45 = sub_1648A40(v10);
    if ( v54 != v45 )
    {
      v52 = v10;
      v46 = v31;
      v47 = v45;
      do
      {
        v48 = v11[1];
        v49 = *v11;
        v47 += 16;
        v11 += 7;
        v50 = sub_16052C0(v42, v49, v48);
        *(_DWORD *)(v47 - 8) = v46;
        *(_QWORD *)(v47 - 16) = v50;
        v46 += (*(v11 - 2) - *(v11 - 3)) >> 3;
        *(_DWORD *)(v47 - 4) = v46;
      }
      while ( v54 != v47 );
      v10 = v52;
    }
  }
  return sub_164B780(v10, a6);
}
