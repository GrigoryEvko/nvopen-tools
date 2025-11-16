// Function: sub_2C7AA20
// Address: 0x2c7aa20
//
void __fastcall sub_2C7AA20(__int64 a1, const char *a2)
{
  __int64 v2; // r10
  unsigned int v5; // esi
  int v6; // r11d
  __int64 v7; // r8
  const char **v8; // rdi
  unsigned int v9; // ecx
  const char **v10; // rdx
  const char *v11; // rax
  int v12; // eax
  int v13; // edx
  unsigned __int8 v14; // al
  int v15; // ecx
  __int64 v16; // r14
  __int64 v17; // rbx
  const char *i; // rax
  unsigned __int8 *v19; // rsi
  int v20; // edx
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rdi
  _WORD *v25; // rdx
  char *v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rdi
  __int64 v31; // rdx
  __m128i si128; // xmm0
  _BYTE *v33; // rax
  _BYTE *v34; // rax
  __int64 v35; // rdi
  int v36; // eax
  int v37; // ecx
  __int64 v38; // r8
  unsigned int v39; // eax
  const char *v40; // rsi
  int v41; // r10d
  const char **v42; // r9
  int v43; // eax
  int v44; // eax
  __int64 v45; // rsi
  int v46; // r9d
  const char **v47; // r8
  unsigned int v48; // ebx
  const char *v49; // rcx

  v2 = a1 + 152;
  v5 = *(_DWORD *)(a1 + 176);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 152);
    goto LABEL_49;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 160);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (const char **)(v7 + 8LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    return;
  while ( v11 != (const char *)-4096LL )
  {
    if ( v11 != (const char *)-8192LL || v8 )
      v10 = v8;
    v9 = (v5 - 1) & (v6 + v9);
    v11 = *(const char **)(v7 + 8LL * v9);
    if ( a2 == v11 )
      return;
    ++v6;
    v8 = v10;
    v10 = (const char **)(v7 + 8LL * v9);
  }
  v12 = *(_DWORD *)(a1 + 168);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)(a1 + 152);
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v5 )
  {
LABEL_49:
    sub_BD1680(v2, 2 * v5);
    v36 = *(_DWORD *)(a1 + 176);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 160);
      v39 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (const char **)(v38 + 8LL * v39);
      v13 = *(_DWORD *)(a1 + 168) + 1;
      v40 = *v8;
      if ( a2 != *v8 )
      {
        v41 = 1;
        v42 = 0;
        while ( v40 != (const char *)-4096LL )
        {
          if ( v40 == (const char *)-8192LL && !v42 )
            v42 = v8;
          v39 = v37 & (v41 + v39);
          v8 = (const char **)(v38 + 8LL * v39);
          v40 = *v8;
          if ( a2 == *v8 )
            goto LABEL_13;
          ++v41;
        }
        if ( v42 )
          v8 = v42;
      }
      goto LABEL_13;
    }
    goto LABEL_78;
  }
  if ( v5 - *(_DWORD *)(a1 + 172) - v13 <= v5 >> 3 )
  {
    sub_BD1680(v2, v5);
    v43 = *(_DWORD *)(a1 + 176);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a1 + 160);
      v46 = 1;
      v47 = 0;
      v48 = v44 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (const char **)(v45 + 8LL * v48);
      v49 = *v8;
      v13 = *(_DWORD *)(a1 + 168) + 1;
      if ( a2 != *v8 )
      {
        while ( v49 != (const char *)-4096LL )
        {
          if ( !v47 && v49 == (const char *)-8192LL )
            v47 = v8;
          v48 = v44 & (v46 + v48);
          v8 = (const char **)(v45 + 8LL * v48);
          v49 = *v8;
          if ( a2 == *v8 )
            goto LABEL_13;
          ++v46;
        }
        if ( v47 )
          v8 = v47;
      }
      goto LABEL_13;
    }
LABEL_78:
    ++*(_DWORD *)(a1 + 168);
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 168) = v13;
  if ( *v8 != (const char *)-4096LL )
    --*(_DWORD *)(a1 + 172);
  *v8 = a2;
  v14 = *(a2 - 16);
  if ( (v14 & 2) != 0 )
    v15 = *((_DWORD *)a2 - 6);
  else
    v15 = (*((_WORD *)a2 - 8) >> 6) & 0xF;
  if ( v15 )
  {
    v16 = 0;
    v17 = 8LL * (unsigned int)(v15 - 1);
    if ( (*(a2 - 16) & 2) == 0 )
      goto LABEL_26;
LABEL_19:
    for ( i = (const char *)*((_QWORD *)a2 - 4); ; i = &a2[-16 - 8LL * ((v14 >> 2) & 0xF)] )
    {
      v19 = *(unsigned __int8 **)&i[v16];
      if ( v19 )
      {
        v20 = *v19;
        if ( (unsigned int)(v20 - 1) > 1 )
        {
          if ( (unsigned __int8)(v20 - 5) > 0x1Fu )
          {
            if ( (_BYTE)v20 != 3 && (_BYTE)v20 )
            {
              v22 = *(_QWORD *)(a1 + 24);
              v23 = *(_QWORD *)(v22 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v22 + 24) - v23) <= 6 )
              {
                sub_CB6200(v22, (unsigned __int8 *)"Error: ", 7u);
              }
              else
              {
                *(_DWORD *)v23 = 1869771333;
                *(_WORD *)(v23 + 4) = 14962;
                *(_BYTE *)(v23 + 6) = 32;
                *(_QWORD *)(v22 + 32) += 7LL;
              }
              v24 = *(_QWORD *)(a1 + 24);
              v25 = *(_WORD **)(v24 + 32);
              if ( *(_QWORD *)(v24 + 24) - (_QWORD)v25 <= 1u )
              {
                sub_CB6200(v24, (unsigned __int8 *)": ", 2u);
              }
              else
              {
                *v25 = 8250;
                *(_QWORD *)(v24 + 32) += 2LL;
              }
              v26 = *(char **)(a1 + 24);
              sub_A61DE0(a2, (__int64)v26, 0);
              v27 = *(_QWORD *)(a1 + 24);
              v28 = *(_QWORD *)(v27 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v27 + 24) - v28) <= 2 )
              {
                v26 = "\n  ";
                sub_CB6200(v27, "\n  ", 3u);
              }
              else
              {
                v29 = 8202;
                *(_BYTE *)(v28 + 2) = 32;
                *(_WORD *)v28 = 8202;
                *(_QWORD *)(v27 + 32) += 3LL;
              }
              v30 = *(_QWORD *)(a1 + 24);
              v31 = *(_QWORD *)(v30 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v30 + 24) - v31) <= 0x14 )
              {
                v26 = "Invalid metadata type";
                v30 = sub_CB6200(v30, "Invalid metadata type", 0x15u);
                v33 = *(_BYTE **)(v30 + 32);
              }
              else
              {
                si128 = _mm_load_si128((const __m128i *)&xmmword_42D0AE0);
                *(_DWORD *)(v31 + 16) = 1887007776;
                *(_BYTE *)(v31 + 20) = 101;
                *(__m128i *)v31 = si128;
                v33 = (_BYTE *)(*(_QWORD *)(v30 + 32) + 21LL);
                *(_QWORD *)(v30 + 32) = v33;
              }
              if ( *(_BYTE **)(v30 + 24) == v33 )
              {
                v26 = "\n";
                sub_CB6200(v30, (unsigned __int8 *)"\n", 1u);
              }
              else
              {
                *v33 = 10;
                ++*(_QWORD *)(v30 + 32);
              }
              v34 = *(_BYTE **)(a1 + 16);
              if ( v34 )
                *v34 = 0;
              if ( !*(_DWORD *)(a1 + 4) )
              {
                v35 = *(_QWORD *)(a1 + 24);
                if ( *(_QWORD *)(v35 + 32) != *(_QWORD *)(v35 + 16) )
                {
                  sub_CB5AE0((__int64 *)v35);
                  v35 = *(_QWORD *)(a1 + 24);
                }
                sub_CEB520(*(_QWORD **)(v35 + 48), (__int64)v26, v31, (char *)v29);
              }
            }
          }
          else
          {
            sub_2C7AA20(a1);
          }
        }
        else
        {
          v21 = *((_QWORD *)v19 + 17);
          if ( v21 )
            sub_2C795F0(a1, v21);
        }
      }
      if ( v17 == v16 )
        break;
      v14 = *(a2 - 16);
      v16 += 8;
      if ( (v14 & 2) != 0 )
        goto LABEL_19;
LABEL_26:
      ;
    }
  }
}
