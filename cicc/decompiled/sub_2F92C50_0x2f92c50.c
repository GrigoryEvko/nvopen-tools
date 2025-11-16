// Function: sub_2F92C50
// Address: 0x2f92c50
//
__int64 __fastcall sub_2F92C50(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v6; // r8
  __int64 v8; // rcx
  int v9; // eax
  unsigned __int64 v10; // rax
  __int64 v11; // rsi
  _QWORD *v12; // r13
  _QWORD *v13; // r15
  __int64 v14; // r8
  __int64 v15; // r14
  unsigned __int64 v16; // rdx
  unsigned int v17; // ebx
  _DWORD *v18; // rsi
  unsigned int v19; // edx
  _BYTE *v20; // r10
  unsigned int v21; // ecx
  __int64 v22; // rax
  unsigned int v23; // edi
  _DWORD *v24; // rsi
  _DWORD *v25; // r11
  unsigned int v26; // edi
  _DWORD *v27; // rsi
  _DWORD *v28; // rsi
  __int64 v29; // rax
  unsigned int v30; // ecx
  __int64 v31; // rsi
  _BYTE *v32; // r8
  unsigned int v33; // eax
  __int64 v34; // rdi
  unsigned int *v35; // rdx
  _BYTE *v37; // rax
  _BYTE *v38; // rdi
  int v39; // edx
  unsigned int v40; // r9d
  __int64 v41; // rdi
  unsigned int v42; // esi
  unsigned int v43; // edx
  __int64 v44; // rax
  _DWORD *v45; // rcx
  _BYTE *v46; // r14
  unsigned int v47; // edx
  _DWORD *v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned int v61; // [rsp+Ch] [rbp-64h]
  unsigned int v62; // [rsp+10h] [rbp-60h]
  unsigned int v63; // [rsp+18h] [rbp-58h]
  unsigned int v64; // [rsp+18h] [rbp-58h]
  unsigned int v65; // [rsp+18h] [rbp-58h]
  _DWORD *v66; // [rsp+20h] [rbp-50h]
  unsigned int v67; // [rsp+28h] [rbp-48h]
  unsigned int v68; // [rsp+2Ch] [rbp-44h]

  v6 = a2;
  v67 = a2[50];
  *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 8LL * v67 + 4) = v67;
  v8 = *(_QWORD *)a2;
  v9 = *(unsigned __int16 *)(*(_QWORD *)a2 + 68LL);
  v68 = (_WORD)v9
     && ((v10 = (unsigned int)(v9 - 9), (unsigned __int16)v10 > 0x3Bu)
      || (v11 = 0x800000000000C09LL, !_bittest64(&v11, v10)))
     && (*(_QWORD *)(*(_QWORD *)(v8 + 16) + 24LL) & 0x10LL) == 0;
  v12 = (_QWORD *)*((_QWORD *)v6 + 5);
  v13 = &v12[2 * (unsigned int)v6[12]];
  if ( v12 == v13 )
    goto LABEL_28;
  v66 = v6;
  v14 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 8LL * v67);
  do
  {
    if ( (*v12 & 6) != 0 )
      goto LABEL_26;
    v15 = *(unsigned int *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 200);
    v16 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
    v17 = *(_DWORD *)(v16 + 200);
    v18 = (_DWORD *)(8 * v15 + *(_QWORD *)(*(_QWORD *)a1 + 8LL));
    if ( (unsigned int)(v14 - *v18) < *(_DWORD *)(*(_QWORD *)a1 + 4LL) )
    {
      if ( (_DWORD)v15 != v18[1] )
        goto LABEL_10;
      v37 = *(_BYTE **)(v16 + 120);
      v38 = &v37[16 * *(unsigned int *)(v16 + 128)];
      if ( v37 == v38 )
      {
LABEL_40:
        v62 = v14;
        v40 = v66[50];
        v18[1] = v40;
        sub_31571F0(a1 + 8, v40, (unsigned int)v15);
        v14 = v62;
        if ( (_DWORD)v15 != *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 8 * v15 + 4) )
        {
LABEL_10:
          v19 = *(_DWORD *)(a1 + 96);
          v20 = (_BYTE *)(v15 + *(_QWORD *)(a1 + 200));
          v21 = (unsigned __int8)*v20;
          if ( v21 < v19 )
          {
            v22 = *(_QWORD *)(a1 + 88);
            v23 = (unsigned __int8)*v20;
            while ( 1 )
            {
              v24 = (_DWORD *)(v22 + 12LL * v23);
              if ( (_DWORD)v15 == *v24 )
                break;
              v23 += 256;
              if ( v19 <= v23 )
                goto LABEL_26;
            }
            a6 = v19;
            v25 = (_DWORD *)(v22 + 12LL * v19);
            v26 = (unsigned __int8)*v20;
            if ( v24 != v25 )
            {
              while ( 1 )
              {
                v27 = (_DWORD *)(v22 + 12LL * v26);
                if ( (_DWORD)v15 == *v27 )
                  break;
                v26 += 256;
                if ( v19 <= v26 )
                  goto LABEL_52;
              }
              if ( v27 != v25 )
              {
                v68 += v27[2];
                goto LABEL_21;
              }
LABEL_52:
              *v20 = v19;
              v49 = *(unsigned int *)(a1 + 96);
              if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 100) )
              {
                v64 = v14;
                sub_C8D5F0(a1 + 88, (const void *)(a1 + 104), v49 + 1, 0xCu, v14, v19);
                v49 = *(unsigned int *)(a1 + 96);
                v14 = v64;
              }
              v50 = *(_QWORD *)(a1 + 88) + 12 * v49;
              *(_QWORD *)v50 = v15 | 0xFFFFFFFF00000000LL;
              *(_DWORD *)(v50 + 8) = 0;
              a6 = (unsigned int)(*(_DWORD *)(a1 + 96) + 1);
              v51 = *(_QWORD *)(a1 + 200);
              *(_DWORD *)(a1 + 96) = a6;
              v19 = a6;
              v21 = *(unsigned __int8 *)(v51 + v15);
              v22 = *(_QWORD *)(a1 + 88);
              v68 += *(_DWORD *)(v22 + 12 * a6 - 4);
              if ( v21 < (unsigned int)a6 )
              {
LABEL_21:
                while ( 1 )
                {
                  v28 = (_DWORD *)(v22 + 12LL * v21);
                  if ( (_DWORD)v15 == *v28 )
                    break;
                  v21 += 256;
                  if ( v21 >= v19 )
                    goto LABEL_26;
                }
                if ( v28 != (_DWORD *)(v22 + 12 * a6) )
                {
                  v29 = v22 + 12 * a6 - 12;
                  if ( v28 != (_DWORD *)v29 )
                  {
                    *(_QWORD *)v28 = *(_QWORD *)v29;
                    v28[2] = *(_DWORD *)(v29 + 8);
                    *(_BYTE *)(*(_QWORD *)(a1 + 200)
                             + *(unsigned int *)(*(_QWORD *)(a1 + 88) + 12LL * *(unsigned int *)(a1 + 96) - 12)) = -85 * (((__int64)v28 - *(_QWORD *)(a1 + 88)) >> 2);
                    v19 = *(_DWORD *)(a1 + 96);
                  }
                  *(_DWORD *)(a1 + 96) = v19 - 1;
                }
              }
            }
          }
          goto LABEL_26;
        }
      }
      else
      {
        v39 = 0;
        while ( (*v37 & 6) != 0 || (unsigned int)++v39 <= 3 )
        {
          v37 += 16;
          if ( v38 == v37 )
            goto LABEL_40;
        }
      }
    }
    else if ( (_DWORD)v15 != v18[1] )
    {
      goto LABEL_10;
    }
    v41 = *(_QWORD *)(a1 + 200);
    v42 = *(_DWORD *)(a1 + 96);
    a6 = v41 + v15;
    v43 = *(unsigned __int8 *)(v41 + v15);
    if ( v43 >= v42 )
      goto LABEL_57;
    v44 = *(_QWORD *)(a1 + 88);
    while ( 1 )
    {
      v45 = (_DWORD *)(v44 + 12LL * v43);
      if ( (_DWORD)v15 == *v45 )
        break;
      v43 += 256;
      if ( v42 <= v43 )
        goto LABEL_57;
    }
    if ( v45 == (_DWORD *)(v44 + 12LL * v42) )
    {
LABEL_57:
      *(_BYTE *)a6 = v42;
      v52 = *(unsigned int *)(a1 + 96);
      if ( v52 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 100) )
      {
        v65 = v14;
        sub_C8D5F0(a1 + 88, (const void *)(a1 + 104), v52 + 1, 0xCu, v14, a6);
        v52 = *(unsigned int *)(a1 + 96);
        v14 = v65;
      }
      v53 = *(_QWORD *)(a1 + 88) + 12 * v52;
      *(_QWORD *)v53 = v15 | 0xFFFFFFFF00000000LL;
      *(_DWORD *)(v53 + 8) = 0;
      v54 = (unsigned int)(*(_DWORD *)(a1 + 96) + 1);
      v44 = *(_QWORD *)(a1 + 88);
      *(_DWORD *)(a1 + 96) = v54;
      v42 = v54;
      if ( *(_DWORD *)(v44 + 12 * v54 - 8) == -1 )
      {
        v46 = (_BYTE *)(*(_QWORD *)(a1 + 200) + v15);
        v47 = (unsigned __int8)*v46;
        a6 = (unsigned int)v66[50];
        if ( v47 >= v42 )
          goto LABEL_65;
LABEL_49:
        while ( 1 )
        {
          v48 = (_DWORD *)(v44 + 12LL * v47);
          if ( v17 == *v48 )
            break;
          v47 += 256;
          if ( v47 >= v42 )
            goto LABEL_65;
        }
        if ( v48 == (_DWORD *)(v44 + 12LL * v42) )
        {
LABEL_65:
          *v46 = v42;
          v58 = *(unsigned int *)(a1 + 96);
          if ( v58 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 100) )
          {
            v61 = v14;
            v63 = a6;
            sub_C8D5F0(a1 + 88, (const void *)(a1 + 104), v58 + 1, 0xCu, v14, a6);
            v58 = *(unsigned int *)(a1 + 96);
            v14 = v61;
            a6 = v63;
          }
          v59 = *(_QWORD *)(a1 + 88) + 12 * v58;
          *(_QWORD *)v59 = v17 | 0xFFFFFFFF00000000LL;
          *(_DWORD *)(v59 + 8) = 0;
          v60 = (unsigned int)(*(_DWORD *)(a1 + 96) + 1);
          *(_DWORD *)(a1 + 96) = v60;
          v48 = (_DWORD *)(*(_QWORD *)(a1 + 88) + 12 * v60 - 12);
        }
        v48[1] = a6;
      }
    }
    else if ( v45[1] == -1 )
    {
      v46 = (_BYTE *)(v41 + v15);
      v47 = (unsigned __int8)*v46;
      a6 = (unsigned int)v66[50];
      goto LABEL_49;
    }
LABEL_26:
    v12 += 2;
  }
  while ( v13 != v12 );
  v6 = v66;
LABEL_28:
  v30 = *(_DWORD *)(a1 + 96);
  v31 = (unsigned int)v6[50];
  v32 = (_BYTE *)(*(_QWORD *)(a1 + 200) + v31);
  v33 = (unsigned __int8)*v32;
  if ( v33 >= v30 )
    goto LABEL_62;
  v34 = *(_QWORD *)(a1 + 88);
  while ( 1 )
  {
    v35 = (unsigned int *)(v34 + 12LL * v33);
    if ( (_DWORD)v31 == *v35 )
      break;
    v33 += 256;
    if ( v30 <= v33 )
      goto LABEL_62;
  }
  if ( v35 == (unsigned int *)(v34 + 12LL * v30) )
  {
LABEL_62:
    *v32 = v30;
    v55 = *(unsigned int *)(a1 + 96);
    if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 100) )
    {
      sub_C8D5F0(a1 + 88, (const void *)(a1 + 104), v55 + 1, 0xCu, (__int64)v32, a6);
      v55 = *(unsigned int *)(a1 + 96);
    }
    v56 = *(_QWORD *)(a1 + 88) + 12 * v55;
    *(_QWORD *)v56 = v31 | 0xFFFFFFFF00000000LL;
    *(_DWORD *)(v56 + 8) = 0;
    v57 = (unsigned int)(*(_DWORD *)(a1 + 96) + 1);
    *(_DWORD *)(a1 + 96) = v57;
    v35 = (unsigned int *)(*(_QWORD *)(a1 + 88) + 12 * v57 - 12);
  }
  v35[1] = -1;
  *v35 = v67;
  v35[2] = v68;
  return v68;
}
