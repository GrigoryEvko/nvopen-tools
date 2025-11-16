// Function: sub_1515740
// Address: 0x1515740
//
__int64 *__fastcall sub_1515740(__int64 *a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // r14
  _QWORD *v8; // r15
  __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  _BYTE *v11; // r8
  unsigned __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r11
  __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  _BYTE *v22; // r8
  unsigned int v23; // eax
  unsigned int v24; // eax
  unsigned int v25; // edx
  __int64 v26; // rsi
  unsigned int v27; // edx
  __int64 v28; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r11
  __int64 v33; // rdx
  unsigned __int64 v34; // rcx
  _QWORD *v35; // rdx
  unsigned __int64 v36; // rcx
  char *v37; // rdx
  _BYTE *v38; // r8
  unsigned int v39; // eax
  unsigned int v40; // eax
  unsigned int v41; // edx
  __int64 v42; // rsi
  unsigned int v43; // r11d
  unsigned __int64 v44; // rcx
  unsigned __int64 v45; // rcx
  char *v46; // rdx
  _BYTE *v47; // r8
  unsigned int v48; // eax
  unsigned int v49; // eax
  unsigned int v50; // edx
  __int64 v51; // rsi
  _QWORD *v52; // rdi
  _QWORD *v53; // rax
  char *v54; // r14
  unsigned __int64 v55; // rax
  __int64 v56; // [rsp+0h] [rbp-70h]
  __int64 v57; // [rsp+0h] [rbp-70h]
  __int64 v58; // [rsp+0h] [rbp-70h]
  _BYTE *v59; // [rsp+8h] [rbp-68h]
  _BYTE *v60; // [rsp+8h] [rbp-68h]
  _BYTE *v61; // [rsp+8h] [rbp-68h]
  __int64 v62; // [rsp+10h] [rbp-60h]
  __int64 v63; // [rsp+10h] [rbp-60h]
  __int64 v64; // [rsp+10h] [rbp-60h]
  __int64 v65; // [rsp+10h] [rbp-60h]
  __int64 v66; // [rsp+10h] [rbp-60h]
  __int64 v67; // [rsp+10h] [rbp-60h]
  const char *v69; // [rsp+20h] [rbp-50h] BYREF
  char v70; // [rsp+30h] [rbp-40h]
  char v71; // [rsp+31h] [rbp-3Fh]

  v7 = *(_QWORD *)(a4 + 8);
  if ( a3 == 2 )
  {
LABEL_6:
    v8 = *(_QWORD **)a4;
    v9 = *(unsigned int *)(a5 + 8);
    if ( !v7 )
    {
LABEL_26:
      *(_QWORD *)a4 = *(_QWORD *)a5;
      *(_QWORD *)(a4 + 8) = (unsigned int)v9;
      goto LABEL_28;
    }
    while ( 1 )
    {
      v10 = *v8;
      v11 = v8 + 1;
      if ( *v8 > 0x22u )
        break;
      v12 = 1;
      if ( v10 > 0xF )
        v12 = 1 - ((((1LL << v10) & 0x410010000LL) == 0) - 1LL);
      v13 = *(_DWORD *)(a5 + 12);
      if ( v12 > v7 )
        v12 = v7;
      v14 = v12;
      v15 = v12 - 1;
      if ( v10 == 28 )
      {
        if ( v13 <= (unsigned int)v9 )
        {
          v67 = v15;
          sub_16CD150(a5, a5 + 16, 0, 8);
          v9 = *(unsigned int *)(a5 + 8);
          v11 = v8 + 1;
          v15 = v67;
        }
        v16 = 8 * v15;
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v9) = 16;
        v17 = v16 >> 3;
        v18 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
        v19 = *(unsigned int *)(a5 + 12) - v18;
        *(_DWORD *)(a5 + 8) = v18;
        if ( v16 >> 3 > v19 )
        {
          v56 = v16;
          v59 = v11;
          v63 = v16 >> 3;
          sub_16CD150(a5, a5 + 16, v17 + v18, 8);
          v18 = *(unsigned int *)(a5 + 8);
          v16 = v56;
          v11 = v59;
          LODWORD(v17) = v63;
        }
        if ( !v16 )
          goto LABEL_23;
        v18 = *(_QWORD *)a5 + 8 * v18;
        if ( (unsigned int)v16 < 8 )
        {
          if ( (_DWORD)v16 )
          {
            *(_BYTE *)v18 = *v11;
            LODWORD(v18) = *(_DWORD *)(a5 + 8);
            goto LABEL_23;
          }
        }
        else
        {
          *(_QWORD *)v18 = *(_QWORD *)v11;
          *(_QWORD *)(v18 + (unsigned int)v16 - 8) = *(_QWORD *)&v11[(unsigned int)v16 - 8];
          v20 = (v18 + 8) & 0xFFFFFFFFFFFFFFF8LL;
          v21 = v18 - v20;
          v22 = &v11[-v21];
          v23 = (v21 + v16) & 0xFFFFFFF8;
          if ( v23 >= 8 )
          {
            v24 = v23 & 0xFFFFFFF8;
            v25 = 0;
            do
            {
              v26 = v25;
              v25 += 8;
              *(_QWORD *)(v20 + v26) = *(_QWORD *)&v22[v26];
            }
            while ( v25 < v24 );
          }
        }
        LODWORD(v18) = *(_DWORD *)(a5 + 8);
LABEL_23:
        v27 = v17 + v18;
        *(_DWORD *)(a5 + 8) = v27;
        v28 = v27;
        if ( *(_DWORD *)(a5 + 12) <= v27 )
        {
          sub_16CD150(a5, a5 + 16, 0, 8);
          v28 = *(unsigned int *)(a5 + 8);
        }
        v8 += v14;
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v28) = 28;
        v9 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
        *(_DWORD *)(a5 + 8) = v9;
        v7 -= v14;
        if ( !v7 )
          goto LABEL_26;
      }
      else
      {
        if ( v10 != 34 )
          goto LABEL_34;
        if ( v13 <= (unsigned int)v9 )
        {
          v66 = v15;
          sub_16CD150(a5, a5 + 16, 0, 8);
          v9 = *(unsigned int *)(a5 + 8);
          v11 = v8 + 1;
          v15 = v66;
        }
        v31 = 8 * v15;
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v9) = 35;
        v32 = v31 >> 3;
        v33 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
        v44 = *(unsigned int *)(a5 + 12) - v33;
        *(_DWORD *)(a5 + 8) = v33;
        if ( v31 >> 3 > v44 )
        {
          v58 = v31;
          v61 = v11;
          v65 = v31 >> 3;
          sub_16CD150(a5, a5 + 16, v32 + v33, 8);
          v33 = *(unsigned int *)(a5 + 8);
          v31 = v58;
          v11 = v61;
          LODWORD(v32) = v65;
        }
        if ( !v31 )
          goto LABEL_43;
        v35 = (_QWORD *)(*(_QWORD *)a5 + 8 * v33);
        if ( (unsigned int)v31 >= 8 )
        {
          *v35 = *(_QWORD *)v11;
          *(_QWORD *)((char *)v35 + (unsigned int)v31 - 8) = *(_QWORD *)&v11[(unsigned int)v31 - 8];
          v45 = (unsigned __int64)(v35 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          v46 = (char *)v35 - v45;
          v47 = (_BYTE *)(v11 - v46);
          v48 = ((_DWORD)v46 + v31) & 0xFFFFFFF8;
          if ( v48 >= 8 )
          {
            v49 = v48 & 0xFFFFFFF8;
            v50 = 0;
            do
            {
              v51 = v50;
              v50 += 8;
              *(_QWORD *)(v45 + v51) = *(_QWORD *)&v47[v51];
            }
            while ( v50 < v49 );
          }
          goto LABEL_42;
        }
LABEL_52:
        if ( (_DWORD)v31 )
          *(_BYTE *)v35 = *v11;
LABEL_42:
        LODWORD(v33) = *(_DWORD *)(a5 + 8);
LABEL_43:
        v43 = v33 + v32;
        v8 += v14;
        *(_DWORD *)(a5 + 8) = v43;
        v9 = v43;
        v7 -= v14;
        if ( !v7 )
        {
          *(_QWORD *)a4 = *(_QWORD *)a5;
          *(_QWORD *)(a4 + 8) = v43;
LABEL_28:
          *a1 = 1;
          return a1;
        }
      }
    }
    if ( v10 == 4096 )
    {
      v30 = 3;
      if ( v7 <= 3 )
        v30 = v7;
      v14 = v30;
      v15 = v30 - 1;
LABEL_34:
      if ( (unsigned int)v9 < *(_DWORD *)(a5 + 12) )
      {
LABEL_35:
        v31 = 8 * v15;
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v9) = v10;
        v32 = v31 >> 3;
        v33 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
        v34 = *(unsigned int *)(a5 + 12) - v33;
        *(_DWORD *)(a5 + 8) = v33;
        if ( v31 >> 3 > v34 )
        {
          v57 = v31;
          v60 = v11;
          v64 = v31 >> 3;
          sub_16CD150(a5, a5 + 16, v32 + v33, 8);
          v33 = *(unsigned int *)(a5 + 8);
          v31 = v57;
          v11 = v60;
          LODWORD(v32) = v64;
        }
        if ( !v31 )
          goto LABEL_43;
        v35 = (_QWORD *)(*(_QWORD *)a5 + 8 * v33);
        if ( (unsigned int)v31 >= 8 )
        {
          *v35 = *(_QWORD *)v11;
          *(_QWORD *)((char *)v35 + (unsigned int)v31 - 8) = *(_QWORD *)&v11[(unsigned int)v31 - 8];
          v36 = (unsigned __int64)(v35 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          v37 = (char *)v35 - v36;
          v38 = (_BYTE *)(v11 - v37);
          v39 = ((_DWORD)v37 + v31) & 0xFFFFFFF8;
          if ( v39 >= 8 )
          {
            v40 = v39 & 0xFFFFFFF8;
            v41 = 0;
            do
            {
              v42 = v41;
              v41 += 8;
              *(_QWORD *)(v36 + v42) = *(_QWORD *)&v38[v42];
            }
            while ( v41 < v40 );
          }
          goto LABEL_42;
        }
        goto LABEL_52;
      }
    }
    else
    {
      v14 = 1;
      v15 = 0;
      if ( (unsigned int)v9 < *(_DWORD *)(a5 + 12) )
        goto LABEL_35;
    }
    v62 = v15;
    sub_16CD150(a5, a5 + 16, 0, 8);
    v9 = *(unsigned int *)(a5 + 8);
    v10 = *v8;
    v11 = v8 + 1;
    v15 = v62;
    goto LABEL_35;
  }
  if ( a3 <= 2 )
  {
    if ( a3 || v7 <= 2 )
    {
      if ( !v7 )
        goto LABEL_5;
    }
    else
    {
      v52 = *(_QWORD **)a4;
      v53 = (_QWORD *)(*(_QWORD *)a4 + 8 * v7 - 24);
      if ( *v53 != 157 )
      {
        if ( *v52 == 6 )
        {
          v54 = (char *)&v52[v7];
          goto LABEL_66;
        }
        goto LABEL_5;
      }
      *v53 = 4096;
    }
    v52 = *(_QWORD **)a4;
    if ( **(_QWORD **)a4 == 6 )
    {
      v55 = *(_QWORD *)(a4 + 8);
      v54 = (char *)&v52[v55];
      if ( v55 <= 2 )
      {
LABEL_72:
        if ( v54 != (char *)(v52 + 1) )
          memmove(v52, v52 + 1, v54 - (char *)(v52 + 1));
        *((_QWORD *)v54 - 1) = 6;
        goto LABEL_5;
      }
LABEL_66:
      if ( *((_QWORD *)v54 - 3) == 4096 )
        v54 -= 24;
      goto LABEL_72;
    }
LABEL_5:
    *(_BYTE *)(a2 + 1011) = 1;
    v7 = *(_QWORD *)(a4 + 8);
    goto LABEL_6;
  }
  if ( a3 == 3 )
    goto LABEL_28;
  v71 = 1;
  v69 = "Invalid record";
  v70 = 3;
  sub_1514BE0(a1, (__int64)&v69);
  return a1;
}
