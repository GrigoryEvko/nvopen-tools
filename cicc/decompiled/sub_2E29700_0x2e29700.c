// Function: sub_2E29700
// Address: 0x2e29700
//
void __fastcall sub_2E29700(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, int *a5)
{
  _QWORD *v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rcx
  unsigned int v9; // eax
  unsigned int v10; // r12d
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // r9
  __int16 *v14; // rax
  __int16 *v15; // r12
  unsigned int v16; // r13d
  unsigned int v17; // esi
  _DWORD *v18; // rax
  _DWORD *v19; // rdx
  int v20; // eax
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  int *v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // rdx
  _WORD *v26; // rax
  _WORD *v27; // rdx
  unsigned __int16 v28; // cx
  __int64 v29; // rdi
  unsigned int v30; // edx
  int v31; // r12d
  _DWORD *v32; // rax
  _DWORD *v33; // rcx
  unsigned __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rcx
  __int64 v37; // rcx
  __int64 v38; // rsi
  __int16 *v39; // rbx
  int v40; // eax
  _QWORD *v41; // [rsp+0h] [rbp-160h]
  __int64 v43; // [rsp+20h] [rbp-140h]
  unsigned __int16 v45; // [rsp+30h] [rbp-130h]
  _WORD *v47; // [rsp+38h] [rbp-128h]
  unsigned int v48; // [rsp+4Ch] [rbp-114h] BYREF
  _BYTE v49[32]; // [rsp+50h] [rbp-110h] BYREF
  _BYTE *v50; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v51; // [rsp+78h] [rbp-E8h]
  _BYTE v52[136]; // [rsp+80h] [rbp-E0h] BYREF
  int v53; // [rsp+108h] [rbp-58h] BYREF
  unsigned __int64 v54; // [rsp+110h] [rbp-50h]
  int *v55; // [rsp+118h] [rbp-48h]
  int *v56; // [rsp+120h] [rbp-40h]
  __int64 v57; // [rsp+128h] [rbp-38h]

  v5 = a1;
  v50 = v52;
  v51 = 0x2000000000LL;
  v55 = &v53;
  v56 = &v53;
  v6 = a1[12];
  v7 = a1[13];
  v53 = 0;
  v54 = 0;
  v57 = 0;
  v43 = 24LL * a2;
  if ( *(_QWORD *)(v7 + 8LL * a2) || *(_QWORD *)(a1[16] + 8LL * a2) )
  {
    v8 = *(_QWORD *)(v6 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v6 + 8) + v43 + 4);
    v9 = (unsigned __int16)a2;
    if ( v8 )
    {
      v10 = a2;
      v11 = *(_QWORD *)(v6 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v6 + 8) + v43 + 4);
      while ( 1 )
      {
        v11 += 2;
        v48 = v9;
        sub_2E282C0((__int64)v49, (__int64)&v50, &v48, v8, (__int64)a5);
        if ( !*(_WORD *)(v11 - 2) )
          break;
        v10 += *(__int16 *)(v11 - 2);
        v9 = (unsigned __int16)v10;
      }
      v5 = a1;
    }
  }
  else
  {
    v26 = (_WORD *)(*(_QWORD *)(v6 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v6 + 8) + v43 + 4));
    v27 = v26 + 1;
    LOWORD(v26) = *v26;
    v28 = a2 + (_WORD)v26;
    v45 = a2 + (_WORD)v26;
    if ( (_WORD)v26 )
    {
      v29 = v28;
      v47 = v27;
      v30 = v28;
      v41 = v5;
      v31 = v28;
LABEL_35:
      v32 = v50;
      v33 = &v50[4 * (unsigned int)v51];
      if ( v50 != (_BYTE *)v33 )
      {
        while ( v30 != *v32 )
        {
          if ( v33 == ++v32 )
            goto LABEL_49;
        }
        if ( v33 != v32 )
          goto LABEL_40;
      }
LABEL_49:
      if ( *(_QWORD *)(v41[13] + 8 * v29) || *(_QWORD *)(v41[16] + 8 * v29) )
      {
        v37 = v41[12];
        v38 = *(unsigned int *)(*(_QWORD *)(v37 + 8) + 24 * v29 + 4);
        if ( *(_QWORD *)(v37 + 56) + 2 * v38 )
        {
          v39 = (__int16 *)(*(_QWORD *)(v37 + 56) + 2 * v38);
          while ( 1 )
          {
            v48 = v30;
            sub_2E282C0((__int64)v49, (__int64)&v50, &v48, v37, (__int64)a5);
            v40 = *v39++;
            if ( !(_WORD)v40 )
              break;
            v31 += v40;
            v30 = (unsigned __int16)v31;
          }
        }
      }
LABEL_40:
      while ( *v47++ )
      {
        v45 += *(v47 - 1);
        v29 = v45;
        v30 = v45;
        v31 = v45;
        if ( !v57 )
          goto LABEL_35;
        v34 = v54;
        if ( v54 )
        {
          a5 = &v53;
          do
          {
            while ( 1 )
            {
              v35 = *(_QWORD *)(v34 + 16);
              v36 = *(_QWORD *)(v34 + 24);
              if ( (unsigned int)v45 <= *(_DWORD *)(v34 + 32) )
                break;
              v34 = *(_QWORD *)(v34 + 24);
              if ( !v36 )
                goto LABEL_47;
            }
            a5 = (int *)v34;
            v34 = *(_QWORD *)(v34 + 16);
          }
          while ( v35 );
LABEL_47:
          if ( a5 != &v53 && v45 >= (unsigned int)a5[8] )
            continue;
        }
        goto LABEL_49;
      }
      v5 = v41;
    }
  }
  sub_2E285B0(v5, a2, a3);
  v14 = (__int16 *)(*(_QWORD *)(v5[12] + 56LL) + 2LL * *(unsigned int *)(*(_QWORD *)(v5[12] + 8LL) + v43 + 4));
  v15 = v14 + 1;
  LODWORD(v14) = *v14;
  v16 = a2 + (_DWORD)v14;
  if ( (_WORD)v14 )
  {
    v17 = (unsigned __int16)v16;
    if ( v57 )
      goto LABEL_24;
    while ( 1 )
    {
      v18 = v50;
      v19 = &v50[4 * (unsigned int)v51];
      if ( v50 != (_BYTE *)v19 )
      {
        while ( v17 != *v18 )
        {
          if ( v19 == ++v18 )
            goto LABEL_15;
        }
        if ( v19 != v18 )
          break;
      }
      while ( 1 )
      {
LABEL_15:
        v20 = *v15++;
        if ( !(_WORD)v20 )
          goto LABEL_16;
        v16 += v20;
        v17 = (unsigned __int16)v16;
        if ( !v57 )
          break;
LABEL_24:
        v22 = v54;
        if ( v54 )
        {
          v23 = &v53;
          do
          {
            while ( 1 )
            {
              v24 = *(_QWORD *)(v22 + 16);
              v25 = *(_QWORD *)(v22 + 24);
              if ( v17 <= *(_DWORD *)(v22 + 32) )
                break;
              v22 = *(_QWORD *)(v22 + 24);
              if ( !v25 )
                goto LABEL_29;
            }
            v23 = (int *)v22;
            v22 = *(_QWORD *)(v22 + 16);
          }
          while ( v24 );
LABEL_29:
          if ( v23 != &v53 && v17 >= v23[8] )
            goto LABEL_14;
        }
      }
    }
LABEL_14:
    sub_2E285B0(v5, v17, a3);
    goto LABEL_15;
  }
LABEL_16:
  if ( a3 )
  {
    v21 = *(unsigned int *)(a4 + 8);
    if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v21 + 1, 4u, v12, v13);
      v21 = *(unsigned int *)(a4 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a4 + 4 * v21) = a2;
    ++*(_DWORD *)(a4 + 8);
  }
  sub_2E24A60(v54);
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
}
