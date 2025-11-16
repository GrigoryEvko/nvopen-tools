// Function: sub_3252AE0
// Address: 0x3252ae0
//
void __fastcall sub_3252AE0(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  char *v8; // rbx
  char *v9; // r14
  int i; // r12d
  __int64 v11; // rax
  unsigned int v12; // r15d
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  int v15; // ebx
  __int64 v16; // r15
  __int64 v17; // rax
  _DWORD *v18; // r11
  _DWORD *v19; // rcx
  unsigned int *v20; // rsi
  unsigned int *v21; // rdi
  __int64 v22; // r10
  unsigned int *v23; // rdx
  _DWORD *v24; // rax
  __int64 v25; // r13
  int v26; // r12d
  __int64 v27; // rax
  int v28; // ebx
  __int64 v29; // r12
  __int64 v30; // r14
  unsigned int v31; // r15d
  int v32; // esi
  __int64 v33; // rdi
  __int64 v34; // r8
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // r12d
  int v38; // eax
  int v39; // eax
  int v40; // edi
  int v41; // r14d
  int v42; // eax
  __int64 v43; // rcx
  __int64 v44; // rsi
  unsigned int v45; // r14d
  __int64 v46; // r13
  int v47; // eax
  int v48; // edx
  __int64 v49; // rax
  int v50; // [rsp+Ch] [rbp-D4h]
  __int64 v51; // [rsp+10h] [rbp-D0h]
  int v53; // [rsp+20h] [rbp-C0h]
  int v54; // [rsp+24h] [rbp-BCh]
  int v55; // [rsp+24h] [rbp-BCh]
  __int64 v56; // [rsp+28h] [rbp-B8h]
  __int64 *v57; // [rsp+30h] [rbp-B0h]
  int v58; // [rsp+48h] [rbp-98h]
  int v59; // [rsp+48h] [rbp-98h]
  _BYTE *v60; // [rsp+60h] [rbp-80h] BYREF
  __int64 v61; // [rsp+68h] [rbp-78h]
  _BYTE v62[112]; // [rsp+70h] [rbp-70h] BYREF

  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL);
  v8 = *(char **)(v7 + 624);
  v9 = *(char **)(v7 + 616);
  v60 = v62;
  v61 = 0x1000000000LL;
  if ( (unsigned __int64)(v8 - v9) > 0x40 )
  {
    sub_C8D5F0((__int64)&v60, v62, (v8 - v9) >> 2, 4u, a5, a6);
    v9 = *(char **)(v7 + 616);
    v8 = *(char **)(v7 + 624);
  }
  for ( i = -1; v8 != v9; i -= sub_F03EF0(v12) )
  {
    v11 = (unsigned int)v61;
    v12 = *(_DWORD *)v9;
    v13 = (unsigned int)v61 + 1LL;
    if ( v13 > HIDWORD(v61) )
    {
      sub_C8D5F0((__int64)&v60, v62, v13, 4u, a5, a6);
      v11 = (unsigned int)v61;
    }
    v9 += 4;
    *(_DWORD *)&v60[4 * v11] = i;
    LODWORD(v61) = v61 + 1;
  }
  v14 = *((unsigned int *)a2 + 2);
  if ( *(_DWORD *)(a4 + 12) < (unsigned int)v14 )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), v14, 4u, a5, a6);
    v14 = *((unsigned int *)a2 + 2);
  }
  v51 = (__int64)&(*a2)[v14];
  if ( (__int64 *)v51 != *a2 )
  {
    v57 = *a2;
    v15 = 0;
    v16 = a3;
    v56 = 0;
    v50 = 0;
    do
    {
      v17 = v56;
      v18 = *(_DWORD **)(*v57 + 104);
      v19 = *(_DWORD **)(*v57 + 96);
      v56 = *v57;
      if ( v17 )
      {
        v20 = *(unsigned int **)(v17 + 104);
        v21 = *(unsigned int **)(v17 + 96);
        a5 = v18 - v19;
        LODWORD(v22) = a5;
        if ( v20 == v21 )
        {
          if ( v18 != v19 )
          {
            v28 = 0;
            LODWORD(v29) = -1;
            a5 = 0;
            goto LABEL_28;
          }
        }
        else if ( v18 != v19 )
        {
          v23 = *(unsigned int **)(v17 + 96);
          v24 = *(_DWORD **)(*v57 + 96);
          do
          {
            a6 = *v23;
            if ( *v24 != (_DWORD)a6 )
              break;
            ++v24;
            ++v23;
            if ( v24 == v18 )
              break;
          }
          while ( v20 != v23 );
          v25 = v24 - v19;
          if ( (unsigned int)v25 < a5 )
          {
            v28 = v24 - v19;
            a5 = 0;
            LODWORD(v29) = -1;
            if ( (_DWORD)v25 )
            {
              v55 = v20 - v21;
              v29 = (unsigned int)(*(_DWORD *)(v16 + 8) - 1);
              v41 = sub_F03F10(*(int *)(*(_QWORD *)v16 + 12 * v29 + 4));
              v42 = sub_F03F10(*(int *)(*(_QWORD *)v16 + 12 * v29));
              v43 = v29;
              a5 = (unsigned int)(v41 + v42);
              if ( v55 != (_DWORD)v25 )
              {
                v59 = v25;
                v44 = *(_QWORD *)v16;
                v45 = v41 + v42;
                do
                {
                  ++v28;
                  v46 = 12 * v43;
                  v47 = sub_F03F10(*(int *)(v44 + 12 * v43));
                  v44 = *(_QWORD *)v16;
                  v48 = v47;
                  v49 = *(_QWORD *)v16 + v46;
                  v43 = *(unsigned int *)(v49 + 8);
                  v45 -= *(_DWORD *)(v49 + 4) + v48;
                }
                while ( v55 != v28 );
                v28 = v59;
                a5 = v45;
                LODWORD(v29) = *(_DWORD *)(v49 + 8);
              }
              v19 = *(_DWORD **)(v56 + 96);
              v53 = (__int64)(*(_QWORD *)(v56 + 104) - (_QWORD)v19) >> 2;
              if ( v28 == v53 )
              {
LABEL_47:
                v54 = 0;
                v39 = v50;
LABEL_40:
                v40 = v50;
                v50 = v39;
                v26 = v54 + v40 + 1 - a5;
                v15 = v26;
                goto LABEL_20;
              }
LABEL_29:
              v30 = v16;
              v58 = v29;
              v31 = a5;
              v54 = 0;
              while ( 1 )
              {
                v37 = v19[v28];
                if ( v37 < 0 )
                  v37 = *(_DWORD *)&v60[4 * ~v37];
                v38 = sub_F03F10(v37);
                if ( v31 )
                {
                  v32 = -(v38 + v31);
                  v33 = v32;
                }
                else
                {
                  v33 = 0;
                  v32 = 0;
                }
                v31 = sub_F03F10(v33) + v38;
                v54 += v31;
                v35 = *(unsigned int *)(v30 + 8);
                if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(v30 + 12) )
                {
                  sub_C8D5F0(v30, (const void *)(v30 + 16), v35 + 1, 0xCu, v34, a6);
                  v35 = *(unsigned int *)(v30 + 8);
                }
                ++v28;
                v36 = *(_QWORD *)v30 + 12 * v35;
                *(_QWORD *)v36 = __PAIR64__(v32, v37);
                *(_DWORD *)(v36 + 8) = v58;
                v58 = *(_DWORD *)(v30 + 8);
                *(_DWORD *)(v30 + 8) = v58 + 1;
                if ( v53 == v28 )
                  break;
                v19 = *(_DWORD **)(v56 + 96);
              }
              a5 = v31;
              v39 = v50 + v54;
              v16 = v30;
              goto LABEL_40;
            }
LABEL_28:
            v53 = v22;
            if ( v28 == (_DWORD)v22 )
              goto LABEL_47;
            goto LABEL_29;
          }
        }
      }
      else if ( v18 != v19 )
      {
        v28 = 0;
        LODWORD(v29) = -1;
        a5 = 0;
        v22 = v18 - v19;
        goto LABEL_28;
      }
      v26 = v15;
LABEL_20:
      v27 = *(unsigned int *)(a4 + 8);
      if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v27 + 1, 4u, a5, a6);
        v27 = *(unsigned int *)(a4 + 8);
      }
      ++v57;
      *(_DWORD *)(*(_QWORD *)a4 + 4 * v27) = v26;
      ++*(_DWORD *)(a4 + 8);
    }
    while ( (__int64 *)v51 != v57 );
  }
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
}
