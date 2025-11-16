// Function: sub_2B73600
// Address: 0x2b73600
//
int *__fastcall sub_2B73600(unsigned int **a1, int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // esi
  int *v9; // r10
  int *v11; // rdx
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // r14
  signed int v15; // r15d
  int *v16; // r11
  unsigned int v17; // ecx
  _DWORD *v18; // rdx
  int v19; // eax
  bool v20; // zf
  unsigned int v21; // ecx
  unsigned int v22; // edx
  unsigned int v23; // ecx
  int v24; // ebx
  unsigned int v25; // ebx
  unsigned int v26; // r12d
  unsigned int v27; // r9d
  int v28; // edx
  unsigned int v29; // eax
  int v30; // esi
  unsigned int *v31; // r10
  unsigned int v32; // r9d
  int v33; // esi
  unsigned int v34; // eax
  unsigned int v35; // eax
  _DWORD *v36; // rax
  _DWORD *v37; // r9
  _DWORD *v38; // r10
  int *v39; // rsi
  int v40; // ebx
  int v41; // edi
  unsigned int v42; // eax
  __int64 v43; // rax
  int v44; // ebx
  __int64 v45; // [rsp+8h] [rbp-98h]
  __int64 v46; // [rsp+8h] [rbp-98h]
  int *v47; // [rsp+10h] [rbp-90h]
  int v48; // [rsp+10h] [rbp-90h]
  int *v49; // [rsp+10h] [rbp-90h]
  unsigned int v50; // [rsp+18h] [rbp-88h]
  int v51; // [rsp+20h] [rbp-80h]
  __int64 v52; // [rsp+20h] [rbp-80h]
  unsigned int *v53; // [rsp+28h] [rbp-78h]
  _DWORD *v54; // [rsp+28h] [rbp-78h]
  unsigned int v55; // [rsp+28h] [rbp-78h]
  int *v56; // [rsp+28h] [rbp-78h]
  int *v57; // [rsp+28h] [rbp-78h]
  int v59; // [rsp+3Ch] [rbp-64h]
  int *v60; // [rsp+48h] [rbp-58h] BYREF
  __int64 v61; // [rsp+50h] [rbp-50h] BYREF
  __int64 v62; // [rsp+58h] [rbp-48h]
  __int64 v63; // [rsp+60h] [rbp-40h]
  __int64 v64; // [rsp+68h] [rbp-38h]

  v7 = *a1[1];
  if ( **a1 <= v7 )
  {
    BYTE4(v60) = 0;
    return v60;
  }
  v9 = &a2[a3];
  if ( a2 == v9 )
  {
    v12 = 0x7FFFFFFF;
  }
  else
  {
    v11 = a2;
    v12 = 0x7FFFFFFF;
    do
    {
      if ( *v11 < v12 && *v11 != -1 )
        v12 = *v11;
      ++v11;
    }
    while ( v11 != v9 );
  }
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v59 = v7 * (v12 / v7);
  if ( *(_DWORD *)(a4 + 12) )
  {
    if ( !*(_DWORD *)(a4 + 8) || (**(_DWORD **)a4 = v59, !*(_DWORD *)(a4 + 8)) )
      **(_DWORD **)a4 = v59;
    *(_DWORD *)(a4 + 8) = 1;
  }
  else
  {
    *(_DWORD *)(a4 + 8) = 0;
    v56 = v9;
    sub_C8D5F0(a4, (const void *)(a4 + 16), 1u, 4u, a5, a6);
    v9 = v56;
    **(_DWORD **)a4 = v59;
    *(_DWORD *)(a4 + 8) = 1;
  }
  if ( a2 != v9 )
  {
    v13 = (__int64)a2;
    v14 = 0;
    v15 = -1;
    v51 = 7;
    v16 = v9;
    v50 = v59;
    while ( 1 )
    {
      v24 = *(_DWORD *)(v13 + 4 * v14);
      if ( v24 != -1 )
        break;
LABEL_22:
      ++v14;
      if ( v16 == (int *)(v13 + 4 * v14) )
        goto LABEL_38;
    }
    v25 = v24 - v59;
    v26 = *a1[2] * (v25 / **a1) + v25 % **a1 / *a1[1];
    if ( v15 < 0 )
      v15 = *a1[2] * (v25 / **a1) + v25 % **a1 / *a1[1];
    if ( (_DWORD)v64 )
    {
      v17 = (v64 - 1) & (37 * v26);
      v18 = (_DWORD *)(v62 + 4LL * v17);
      v19 = *v18;
      if ( v26 == *v18 )
      {
LABEL_17:
        v20 = (_DWORD)v63 == 2;
        if ( (unsigned int)v63 > 2 )
          goto LABEL_49;
LABEL_18:
        if ( v20 )
        {
          if ( *(_DWORD *)(a4 + 8) == 1 )
          {
            v35 = *a1[1];
            v60 = (int *)v13;
            v55 = v35;
            sub_2B097A0(&v60, v14);
            v39 = v60;
            if ( v16 == v60 )
            {
              v42 = 0x7FFFFFFF;
            }
            else
            {
              v40 = 0x7FFFFFFF;
              do
              {
                v41 = *v39;
                if ( *v39 != -1
                  && *v36 * ((unsigned int)(v41 - v59) / *v38) + (unsigned int)(v41 - v59) % *v38 / *v37 != v15
                  && v41 < v40 )
                {
                  v40 = *v39;
                }
                ++v39;
              }
              while ( v16 != v39 );
              v42 = v40;
            }
            v50 = v55 * (v42 / v55);
            v43 = *(unsigned int *)(a4 + 8);
            v44 = v50 % **a1;
            if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
            {
              v52 = v13;
              v57 = v16;
              sub_C8D5F0(a4, (const void *)(a4 + 16), v43 + 1, 4u, v13, (__int64)v37);
              v13 = v52;
              v16 = v57;
              v43 = *(unsigned int *)(a4 + 8);
            }
            *(_DWORD *)(*(_QWORD *)a4 + 4 * v43) = v44;
            ++*(_DWORD *)(a4 + 8);
          }
          v51 = 6;
          v25 = *(_DWORD *)(v13 + 4 * v14) - v50;
        }
        v21 = *a1[1];
        v22 = v25 % **a1 % v21;
        v23 = v22 + v21;
        if ( v26 == v15 )
          v23 = v22;
        *(_DWORD *)(v13 + 4 * v14) = v23;
        goto LABEL_22;
      }
      v48 = 1;
      v54 = 0;
      while ( v19 != 0x7FFFFFFF )
      {
        if ( v19 != 0x80000000 || v54 )
          v18 = v54;
        v17 = (v64 - 1) & (v48 + v17);
        v19 = *(_DWORD *)(v62 + 4LL * v17);
        if ( v26 == v19 )
          goto LABEL_17;
        v54 = v18;
        v18 = (_DWORD *)(v62 + 4LL * v17);
        ++v48;
      }
      if ( v54 )
        v18 = v54;
      ++v61;
      v53 = v18;
      v28 = v63 + 1;
      if ( 4 * ((int)v63 + 1) < (unsigned int)(3 * v64) )
      {
        if ( (int)v64 - HIDWORD(v63) - v28 <= (unsigned int)v64 >> 3 )
        {
          v46 = v13;
          v49 = v16;
          sub_29F8760((__int64)&v61, v64);
          if ( !(_DWORD)v64 )
          {
LABEL_83:
            LODWORD(v63) = v63 + 1;
            BUG();
          }
          v31 = 0;
          v16 = v49;
          v32 = (v64 - 1) & (37 * v26);
          v13 = v46;
          v28 = v63 + 1;
          v33 = 1;
          v53 = (unsigned int *)(v62 + 4LL * v32);
          v34 = *v53;
          if ( v26 != *v53 )
          {
            while ( v34 != 0x7FFFFFFF )
            {
              if ( !v31 && v34 == 0x80000000 )
                v31 = v53;
              v32 = (v64 - 1) & (v33 + v32);
              v53 = (unsigned int *)(v62 + 4LL * v32);
              v34 = *v53;
              if ( v26 == *v53 )
                goto LABEL_46;
              ++v33;
            }
            goto LABEL_32;
          }
        }
LABEL_46:
        LODWORD(v63) = v28;
        if ( *v53 != 0x7FFFFFFF )
          --HIDWORD(v63);
        *v53 = v26;
        v20 = (_DWORD)v63 == 2;
        if ( (unsigned int)v63 > 2 )
        {
LABEL_49:
          BYTE4(v60) = 0;
          goto LABEL_39;
        }
        goto LABEL_18;
      }
    }
    else
    {
      ++v61;
    }
    v45 = v13;
    v47 = v16;
    sub_29F8760((__int64)&v61, 2 * v64);
    if ( !(_DWORD)v64 )
      goto LABEL_83;
    v16 = v47;
    v13 = v45;
    v27 = (v64 - 1) & (37 * v26);
    v28 = v63 + 1;
    v53 = (unsigned int *)(v62 + 4LL * v27);
    v29 = *v53;
    if ( v26 != *v53 )
    {
      v30 = 1;
      v31 = 0;
      while ( v29 != 0x7FFFFFFF )
      {
        if ( v29 == 0x80000000 && !v31 )
          v31 = v53;
        v27 = (v64 - 1) & (v30 + v27);
        v53 = (unsigned int *)(v62 + 4LL * v27);
        v29 = *v53;
        if ( v26 == *v53 )
          goto LABEL_46;
        ++v30;
      }
LABEL_32:
      if ( !v31 )
        v31 = v53;
      v53 = v31;
      goto LABEL_46;
    }
    goto LABEL_46;
  }
  v51 = 7;
LABEL_38:
  BYTE4(v60) = 1;
  LODWORD(v60) = v51;
LABEL_39:
  sub_C7D6A0(v62, 4LL * (unsigned int)v64, 4);
  return v60;
}
