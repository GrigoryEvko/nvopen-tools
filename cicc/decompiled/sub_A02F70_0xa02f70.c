// Function: sub_A02F70
// Address: 0xa02f70
//
__int64 *__fastcall sub_A02F70(__int64 *a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // r15
  _QWORD *v8; // r8
  __int64 v9; // rdx
  unsigned __int64 v10; // r13
  _BYTE *v11; // r10
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // rcx
  __int64 v15; // rdi
  unsigned __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  unsigned __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  _BYTE *v25; // r10
  unsigned int v26; // eax
  unsigned int v27; // eax
  unsigned int v28; // edx
  __int64 v29; // rsi
  unsigned __int64 v30; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // rsi
  __int64 v34; // r13
  unsigned __int64 v35; // rcx
  __int64 v36; // rax
  _QWORD *v37; // rdx
  unsigned __int64 v38; // rcx
  char *v39; // rdx
  _BYTE *v40; // r10
  unsigned int v41; // r13d
  unsigned int v42; // r13d
  unsigned int v43; // edx
  __int64 v44; // rsi
  unsigned int v45; // eax
  unsigned __int64 v46; // rcx
  unsigned __int64 v47; // rcx
  char *v48; // rdx
  _BYTE *v49; // r10
  unsigned int v50; // r13d
  unsigned int v51; // r13d
  unsigned int v52; // edx
  __int64 v53; // rsi
  _QWORD *v54; // rdi
  _QWORD *v55; // rax
  char *v56; // r13
  unsigned __int64 v57; // rax
  _BYTE *v58; // [rsp+0h] [rbp-80h]
  _BYTE *v59; // [rsp+0h] [rbp-80h]
  int v60; // [rsp+0h] [rbp-80h]
  int v61; // [rsp+0h] [rbp-80h]
  _BYTE *v62; // [rsp+0h] [rbp-80h]
  _BYTE *v63; // [rsp+0h] [rbp-80h]
  __int64 v64; // [rsp+8h] [rbp-78h]
  __int64 v65; // [rsp+8h] [rbp-78h]
  _BYTE *v66; // [rsp+8h] [rbp-78h]
  _BYTE *v67; // [rsp+8h] [rbp-78h]
  __int64 v68; // [rsp+8h] [rbp-78h]
  __int64 v69; // [rsp+8h] [rbp-78h]
  _QWORD *v70; // [rsp+10h] [rbp-70h]
  _QWORD *v71; // [rsp+10h] [rbp-70h]
  _QWORD *v72; // [rsp+10h] [rbp-70h]
  _QWORD *v73; // [rsp+10h] [rbp-70h]
  _QWORD *v74; // [rsp+10h] [rbp-70h]
  _QWORD *v75; // [rsp+10h] [rbp-70h]
  _QWORD *v76; // [rsp+10h] [rbp-70h]
  const char *v78; // [rsp+20h] [rbp-60h] BYREF
  char v79; // [rsp+40h] [rbp-40h]
  char v80; // [rsp+41h] [rbp-3Fh]

  v7 = *(_QWORD *)(a4 + 8);
  if ( a3 == 2 )
  {
LABEL_6:
    v8 = *(_QWORD **)a4;
    LODWORD(v9) = *(_DWORD *)(a5 + 8);
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
      v13 = (unsigned int)v9;
      v14 = *(unsigned int *)(a5 + 12);
      if ( v12 > v7 )
        v12 = v7;
      v15 = (unsigned int)v9;
      v16 = (unsigned int)v9 + 1LL;
      v17 = v12;
      v18 = v12 - 1;
      if ( v10 == 28 )
      {
        if ( v16 > v14 )
        {
          v63 = v8 + 1;
          v69 = v18;
          v75 = v8;
          sub_C8D5F0(a5, a5 + 16, v16, 8);
          v13 = *(unsigned int *)(a5 + 8);
          v11 = v63;
          v18 = v69;
          v8 = v75;
        }
        v19 = 8 * v18;
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v13) = 16;
        v20 = v19 >> 3;
        v21 = *(unsigned int *)(a5 + 12);
        v22 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
        *(_DWORD *)(a5 + 8) = v22;
        if ( (v19 >> 3) + v22 > v21 )
        {
          v59 = v11;
          v65 = v19;
          v71 = v8;
          sub_C8D5F0(a5, a5 + 16, (v19 >> 3) + v22, 8);
          v22 = *(unsigned int *)(a5 + 8);
          v11 = v59;
          v19 = v65;
          v8 = v71;
        }
        if ( !v19 )
          goto LABEL_23;
        v22 = *(_QWORD *)a5 + 8 * v22;
        if ( (unsigned int)v19 < 8 )
        {
          if ( (_DWORD)v19 )
          {
            *(_BYTE *)v22 = *v11;
            LODWORD(v22) = *(_DWORD *)(a5 + 8);
            goto LABEL_23;
          }
        }
        else
        {
          *(_QWORD *)v22 = *(_QWORD *)v11;
          *(_QWORD *)(v22 + (unsigned int)v19 - 8) = *(_QWORD *)&v11[(unsigned int)v19 - 8];
          v23 = (v22 + 8) & 0xFFFFFFFFFFFFFFF8LL;
          v24 = v22 - v23;
          v25 = &v11[-v24];
          v26 = (v24 + v19) & 0xFFFFFFF8;
          if ( v26 >= 8 )
          {
            v27 = v26 & 0xFFFFFFF8;
            v28 = 0;
            do
            {
              v29 = v28;
              v28 += 8;
              *(_QWORD *)(v23 + v29) = *(_QWORD *)&v25[v29];
            }
            while ( v28 < v27 );
          }
        }
        LODWORD(v22) = *(_DWORD *)(a5 + 8);
LABEL_23:
        LODWORD(v9) = v20 + v22;
        v30 = *(unsigned int *)(a5 + 12);
        *(_DWORD *)(a5 + 8) = v9;
        v9 = (unsigned int)v9;
        if ( (unsigned __int64)(unsigned int)v9 + 1 > v30 )
        {
          v76 = v8;
          sub_C8D5F0(a5, a5 + 16, (unsigned int)v9 + 1LL, 8);
          v9 = *(unsigned int *)(a5 + 8);
          v8 = v76;
        }
        v8 += v17;
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v9) = 28;
        LODWORD(v9) = *(_DWORD *)(a5 + 8) + 1;
        *(_DWORD *)(a5 + 8) = v9;
        v7 -= v17;
        if ( !v7 )
          goto LABEL_26;
      }
      else
      {
        if ( v10 != 34 )
          goto LABEL_34;
        if ( v16 > v14 )
        {
          v62 = v8 + 1;
          v68 = v18;
          v74 = v8;
          sub_C8D5F0(a5, a5 + 16, v16, 8);
          v13 = *(unsigned int *)(a5 + 8);
          v11 = v62;
          v18 = v68;
          v8 = v74;
        }
        v34 = 8 * v18;
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v13) = 35;
        v36 = (8 * v18) >> 3;
        v46 = *(unsigned int *)(a5 + 12);
        v9 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
        *(_DWORD *)(a5 + 8) = v9;
        if ( v36 + v9 > v46 )
        {
          v61 = v36;
          v67 = v11;
          v73 = v8;
          sub_C8D5F0(a5, a5 + 16, v36 + v9, 8);
          v9 = *(unsigned int *)(a5 + 8);
          LODWORD(v36) = v61;
          v11 = v67;
          v8 = v73;
        }
        if ( !v34 )
          goto LABEL_43;
        v37 = (_QWORD *)(*(_QWORD *)a5 + 8 * v9);
        if ( (unsigned int)v34 >= 8 )
        {
          *v37 = *(_QWORD *)v11;
          *(_QWORD *)((char *)v37 + (unsigned int)v34 - 8) = *(_QWORD *)&v11[(unsigned int)v34 - 8];
          v47 = (unsigned __int64)(v37 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          v48 = (char *)v37 - v47;
          v49 = (_BYTE *)(v11 - v48);
          v50 = ((_DWORD)v48 + v34) & 0xFFFFFFF8;
          if ( v50 >= 8 )
          {
            v51 = v50 & 0xFFFFFFF8;
            v52 = 0;
            do
            {
              v53 = v52;
              v52 += 8;
              *(_QWORD *)(v47 + v53) = *(_QWORD *)&v49[v53];
            }
            while ( v52 < v51 );
          }
          goto LABEL_42;
        }
LABEL_52:
        if ( (_DWORD)v34 )
          *(_BYTE *)v37 = *v11;
LABEL_42:
        LODWORD(v9) = *(_DWORD *)(a5 + 8);
LABEL_43:
        v45 = v9 + v36;
        v8 += v17;
        *(_DWORD *)(a5 + 8) = v45;
        LODWORD(v9) = v45;
        v7 -= v17;
        if ( !v7 )
        {
          *(_QWORD *)a4 = *(_QWORD *)a5;
          *(_QWORD *)(a4 + 8) = v45;
LABEL_28:
          *a1 = 1;
          return a1;
        }
      }
    }
    if ( v10 == 4096 )
    {
      v32 = 3;
      v14 = *(unsigned int *)(a5 + 12);
      v15 = (unsigned int)v9;
      if ( v7 <= 3 )
        v32 = v7;
      v17 = v32;
      v18 = v32 - 1;
LABEL_34:
      v33 = v15 + 1;
      v9 = v15;
      if ( v15 + 1 <= v14 )
      {
LABEL_35:
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v9) = v10;
        v34 = 8 * v18;
        v35 = *(unsigned int *)(a5 + 12);
        v9 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
        v36 = (8 * v18) >> 3;
        *(_DWORD *)(a5 + 8) = v9;
        if ( v36 + v9 > v35 )
        {
          v60 = v36;
          v66 = v11;
          v72 = v8;
          sub_C8D5F0(a5, a5 + 16, v36 + v9, 8);
          v9 = *(unsigned int *)(a5 + 8);
          LODWORD(v36) = v60;
          v11 = v66;
          v8 = v72;
        }
        if ( !v34 )
          goto LABEL_43;
        v37 = (_QWORD *)(*(_QWORD *)a5 + 8 * v9);
        if ( (unsigned int)v34 >= 8 )
        {
          *v37 = *(_QWORD *)v11;
          *(_QWORD *)((char *)v37 + (unsigned int)v34 - 8) = *(_QWORD *)&v11[(unsigned int)v34 - 8];
          v38 = (unsigned __int64)(v37 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          v39 = (char *)v37 - v38;
          v40 = (_BYTE *)(v11 - v39);
          v41 = ((_DWORD)v39 + v34) & 0xFFFFFFF8;
          if ( v41 >= 8 )
          {
            v42 = v41 & 0xFFFFFFF8;
            v43 = 0;
            do
            {
              v44 = v43;
              v43 += 8;
              *(_QWORD *)(v38 + v44) = *(_QWORD *)&v40[v44];
            }
            while ( v43 < v42 );
          }
          goto LABEL_42;
        }
        goto LABEL_52;
      }
    }
    else
    {
      v17 = 1;
      v18 = 0;
      v33 = (unsigned int)v9 + 1LL;
      v9 = (unsigned int)v9;
      if ( v33 <= *(unsigned int *)(a5 + 12) )
        goto LABEL_35;
    }
    v58 = v8 + 1;
    v64 = v18;
    v70 = v8;
    sub_C8D5F0(a5, a5 + 16, v33, 8);
    v9 = *(unsigned int *)(a5 + 8);
    v11 = v58;
    v18 = v64;
    v8 = v70;
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
      v54 = *(_QWORD **)a4;
      v55 = (_QWORD *)(*(_QWORD *)a4 + 8 * v7 - 24);
      if ( *v55 != 157 )
      {
        if ( *v54 == 6 )
        {
          v56 = (char *)&v54[v7];
          goto LABEL_66;
        }
        goto LABEL_5;
      }
      *v55 = 4096;
    }
    v54 = *(_QWORD **)a4;
    if ( **(_QWORD **)a4 == 6 )
    {
      v57 = *(_QWORD *)(a4 + 8);
      v56 = (char *)&v54[v57];
      if ( v57 <= 2 )
      {
LABEL_72:
        if ( v56 != (char *)(v54 + 1) )
          memmove(v54, v54 + 1, v56 - (char *)(v54 + 1));
        *((_QWORD *)v56 - 1) = 6;
        goto LABEL_5;
      }
LABEL_66:
      if ( *((_QWORD *)v56 - 3) == 4096 )
        v56 -= 24;
      goto LABEL_72;
    }
LABEL_5:
    *(_BYTE *)(a2 + 1099) = 1;
    v7 = *(_QWORD *)(a4 + 8);
    goto LABEL_6;
  }
  if ( a3 == 3 )
    goto LABEL_28;
  v80 = 1;
  v78 = "Invalid record";
  v79 = 3;
  sub_A01DB0(a1, (__int64)&v78);
  return a1;
}
