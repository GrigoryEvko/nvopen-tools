// Function: sub_98E420
// Address: 0x98e420
//
char __fastcall sub_98E420(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, unsigned int a6)
{
  __int64 v9; // r12
  unsigned int *v11; // rax
  __int64 v12; // rcx
  unsigned int v13; // eax
  unsigned __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  int v22; // eax
  __int64 v23; // rcx
  char v24; // al
  unsigned __int8 *v25; // r8
  __int64 v26; // rax
  char v27; // al
  unsigned __int8 *v28; // r8
  __int64 v29; // rax
  __int64 v30; // rax
  _BOOL8 v31; // rsi
  unsigned int v32; // r8d
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rdx
  char *v37; // r10
  char *v38; // rdx
  __int64 v39; // rax
  unsigned int v40; // r12d
  char *v41; // r13
  char *v42; // r10
  _QWORD *v43; // r14
  __int64 v44; // rdx
  unsigned __int64 v45; // rax
  int v46; // edi
  __int64 v47; // r13
  __int64 v48; // rax
  char *v49; // r13
  __int64 v50; // rax
  char *v51; // rbx
  char *v52; // r12
  unsigned int v53; // eax
  __int64 v54; // r13
  __int64 v55; // rax
  unsigned __int8 *v56; // r15
  unsigned __int64 v57; // rax
  char v58; // al
  signed __int64 v59; // rax
  char v60; // al
  __int64 v61; // r8
  signed __int64 v62; // rax
  __int64 v63; // r8
  char v64; // al
  char v65; // al
  char v66; // al
  unsigned int v68; // [rsp+8h] [rbp-88h]
  int v69; // [rsp+8h] [rbp-88h]
  char *v70; // [rsp+10h] [rbp-80h]
  int v71; // [rsp+10h] [rbp-80h]
  __int64 v72; // [rsp+10h] [rbp-80h]
  char *v73; // [rsp+18h] [rbp-78h]
  unsigned int v74; // [rsp+18h] [rbp-78h]
  unsigned int v75; // [rsp+18h] [rbp-78h]
  unsigned __int8 *v76; // [rsp+20h] [rbp-70h]
  int v77; // [rsp+20h] [rbp-70h]
  char *v78; // [rsp+20h] [rbp-70h]
  __int64 v79; // [rsp+20h] [rbp-70h]
  char *v80; // [rsp+20h] [rbp-70h]
  char *v81; // [rsp+20h] [rbp-70h]
  char *v82; // [rsp+20h] [rbp-70h]
  __int64 v83; // [rsp+28h] [rbp-68h]
  int v84; // [rsp+3Ch] [rbp-54h] BYREF
  _QWORD v85[10]; // [rsp+40h] [rbp-50h] BYREF

  v9 = (__int64)a1;
  v83 = a2;
  v11 = (unsigned int *)sub_C94E20(qword_4F862D0);
  if ( v11 )
    v13 = *v11;
  else
    v13 = qword_4F862D0[2];
  if ( a5 >= v13 )
    goto LABEL_17;
  v14 = *a1;
  if ( (_BYTE)v14 == 24 )
    goto LABEL_17;
  if ( (_BYTE)v14 == 22 )
  {
    if ( (unsigned __int8)sub_B2D670(a1, 40) )
      goto LABEL_7;
    if ( (unsigned __int8)sub_B2D670(a1, 90) )
      goto LABEL_7;
    a2 = 91;
    if ( (unsigned __int8)sub_B2D670(a1, 91) )
      goto LABEL_7;
    v14 = *a1;
  }
  if ( (unsigned __int8)v14 <= 0x15u )
  {
    if ( (_BYTE)v14 == 13 )
    {
      LOBYTE(v15) = !(a6 & 1);
      return v15;
    }
    v16 = (unsigned int)(v14 - 12);
    if ( (unsigned __int8)(v14 - 12) <= 1u )
    {
      LOBYTE(v15) = ((a6 >> 1) ^ 1) & 1;
      return v15;
    }
    if ( (_BYTE)v14 == 21 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a1 + 1) + 8LL) - 17 > 1 )
        goto LABEL_23;
    }
    else
    {
      v17 = 1441801;
      if ( _bittest64(&v17, v14) )
        goto LABEL_7;
      v16 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a1 + 1) + 8LL) - 17;
      if ( (unsigned int)v16 > 1 || (_BYTE)v14 == 5 )
        goto LABEL_23;
    }
    if ( ((a6 & 2) == 0 || !(unsigned __int8)sub_AD6C80(a1))
      && ((a6 & 1) == 0 || !(unsigned __int8)sub_AD6C60(a1, a2, v16, v12)) )
    {
      return (unsigned int)sub_AD6CA0(a1) ^ 1;
    }
LABEL_17:
    LOBYTE(v15) = 0;
    return v15;
  }
LABEL_23:
  v18 = *(unsigned __int8 *)sub_BD3E50(a1);
  if ( (unsigned __int8)v18 <= 0x1Cu )
  {
    if ( (unsigned __int8)v18 <= 0x14u )
    {
      v19 = 1048585;
      if ( _bittest64(&v19, v18) )
        goto LABEL_7;
    }
  }
  else if ( (_BYTE)v18 == 60 )
  {
    goto LABEL_7;
  }
  v22 = *a1;
  if ( (unsigned __int8)v22 <= 0x1Cu )
  {
    if ( (_BYTE)v22 != 5 )
      goto LABEL_45;
    goto LABEL_43;
  }
  if ( (_BYTE)v22 == 96 )
    goto LABEL_7;
  if ( (unsigned __int8)(v22 - 34) <= 0x33u )
  {
    v23 = 0x8000000000041LL;
    if ( _bittest64(&v23, (unsigned int)(v22 - 34)) )
    {
      v24 = sub_A74710(a1 + 72, 0, 40);
      v25 = a1 + 72;
      if ( v24 )
        goto LABEL_7;
      v26 = *((_QWORD *)a1 - 4);
      if ( v26 )
      {
        if ( !*(_BYTE *)v26 && *(_QWORD *)(v26 + 24) == *((_QWORD *)a1 + 10) )
        {
          v85[0] = *(_QWORD *)(v26 + 120);
          v58 = sub_A74710(v85, 0, 40);
          v25 = a1 + 72;
          if ( v58 )
            goto LABEL_7;
        }
      }
      v76 = v25;
      v27 = sub_A74710(v25, 0, 90);
      v28 = v76;
      if ( v27 )
        goto LABEL_7;
      v29 = *((_QWORD *)a1 - 4);
      if ( v29 )
      {
        if ( !*(_BYTE *)v29 && *(_QWORD *)(v29 + 24) == *((_QWORD *)a1 + 10) )
        {
          v85[0] = *(_QWORD *)(v29 + 120);
          v60 = sub_A74710(v85, 0, 90);
          v28 = v76;
          if ( v60 )
            goto LABEL_7;
        }
      }
      if ( (unsigned __int8)sub_A74710(v28, 0, 91) )
        goto LABEL_7;
      v30 = *((_QWORD *)a1 - 4);
      if ( v30 )
      {
        if ( !*(_BYTE *)v30 && *(_QWORD *)(v30 + 24) == *((_QWORD *)a1 + 10) )
        {
          v85[0] = *(_QWORD *)(v30 + 120);
          if ( (unsigned __int8)sub_A74710(v85, 0, 91) )
            goto LABEL_7;
        }
      }
      LOBYTE(v22) = *a1;
    }
  }
  if ( (_BYTE)v22 != 84 )
  {
LABEL_43:
    if ( !sub_9860E0(a1, a6, 1) )
    {
      v37 = (char *)sub_986550((__int64)a1);
      v70 = v38;
      v39 = (v38 - v37) >> 7;
      if ( v39 <= 0 )
        goto LABEL_138;
      v68 = a5;
      v40 = a5 + 1;
      v41 = v37;
      v78 = &v37[128 * v39];
      do
      {
        if ( !(unsigned __int8)sub_98E420(*(_QWORD *)v41, v83, a3, a4, v40, a6) )
        {
          v9 = (__int64)a1;
          v42 = v41;
          goto LABEL_80;
        }
        if ( !(unsigned __int8)sub_98E420(*((_QWORD *)v41 + 4), v83, a3, a4, v40, a6) )
        {
          v9 = (__int64)a1;
          v42 = v41 + 32;
          goto LABEL_80;
        }
        if ( !(unsigned __int8)sub_98E420(*((_QWORD *)v41 + 8), v83, a3, a4, v40, a6) )
        {
          v9 = (__int64)a1;
          v42 = v41 + 64;
          goto LABEL_80;
        }
        if ( !(unsigned __int8)sub_98E420(*((_QWORD *)v41 + 12), v83, a3, a4, v40, a6) )
        {
          v9 = (__int64)a1;
          v42 = v41 + 96;
          goto LABEL_80;
        }
        v41 += 128;
      }
      while ( v78 != v41 );
      v37 = v41;
      v9 = (__int64)a1;
      a5 = v68;
LABEL_138:
      v61 = a5 + 1;
      v62 = v70 - v37;
      if ( v70 - v37 == 64 )
        goto LABEL_154;
      if ( v62 != 96 )
      {
        v63 = a5 + 1;
        if ( v62 == 32 )
          goto LABEL_141;
        goto LABEL_7;
      }
      v81 = v37;
      v65 = sub_98E420(*(_QWORD *)v37, v83, a3, a4, a5 + 1, a6);
      v42 = v81;
      if ( v65 )
      {
        v61 = a5 + 1;
        v37 = v81 + 32;
LABEL_154:
        v82 = v37;
        v75 = v61;
        v66 = sub_98E420(*(_QWORD *)v37, v83, a3, a4, v61, a6);
        v42 = v82;
        if ( v66 )
        {
          v63 = v75;
          v37 = v82 + 32;
LABEL_141:
          v80 = v37;
          v64 = sub_98E420(*(_QWORD *)v37, v83, a3, a4, v63, a6);
          v42 = v80;
          if ( v64 )
          {
LABEL_7:
            LOBYTE(v15) = 1;
            return v15;
          }
        }
      }
LABEL_80:
      if ( v70 == v42 )
        goto LABEL_7;
    }
    goto LABEL_44;
  }
  if ( (*((_DWORD *)a1 + 1) & 0x7FFFFFF) == 0 )
    goto LABEL_7;
  v72 = a3;
  v79 = 8LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
  v53 = a5 + 1;
  v54 = 0;
  v74 = v53;
  while ( 1 )
  {
    v55 = *((_QWORD *)a1 - 1);
    v56 = *(unsigned __int8 **)(v55 + 4 * v54);
    if ( !v56 || v56 != a1 )
    {
      v57 = sub_986580(*(_QWORD *)(32LL * *((unsigned int *)a1 + 18) + v55 + v54));
      if ( !(unsigned __int8)sub_98E420(v56, v83, v57, a4, v74, a6) )
        break;
    }
    v54 += 8;
    if ( v54 == v79 )
      goto LABEL_7;
  }
  a3 = v72;
LABEL_44:
  if ( *(_BYTE *)v9 == 61
    && (*(_BYTE *)(v9 + 7) & 0x20) != 0
    && (sub_B91C10(v9, 29)
     || (*(_BYTE *)(v9 + 7) & 0x20) != 0
     && (sub_B91C10(v9, 12) || (*(_BYTE *)(v9 + 7) & 0x20) != 0 && sub_B91C10(v9, 13))) )
  {
    goto LABEL_7;
  }
LABEL_45:
  v77 = a6 & 2;
  v31 = v77 == 0;
  LOBYTE(v15) = sub_98DB10(v9, v31, (unsigned __int8 *)v19, v20, v21);
  if ( (_BYTE)v15 )
    goto LABEL_7;
  if ( a3 )
  {
    v33 = *(_QWORD *)(a3 + 40);
    if ( v33 )
    {
      if ( a4 )
      {
        v34 = (unsigned int)(*(_DWORD *)(v33 + 44) + 1);
        if ( (unsigned int)v34 < *(_DWORD *)(a4 + 32) )
        {
          v35 = *(_QWORD *)(a4 + 24);
          v36 = *(_QWORD *)(v35 + 8 * v34);
          if ( v36 )
          {
            if ( (a6 & 2) != 0 && *(_BYTE *)(*(_QWORD *)(v9 + 8) + 8LL) != 12 || !*(_QWORD *)(v36 + 8) )
            {
LABEL_53:
              v84 = 40;
              sub_CF9770((unsigned int)v85, v9, (unsigned int)&v84, 1, a3, a4, v83);
              LOBYTE(v15) = LODWORD(v85[0]) != 0;
              return v15;
            }
            v71 = a3;
            a3 = v9;
            v69 = a4;
            v43 = *(_QWORD **)(v36 + 8);
            while ( 1 )
            {
              v45 = *(_QWORD *)(*v43 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v45 == *v43 + 48LL )
                goto LABEL_85;
              if ( !v45 )
                BUG();
              v46 = *(unsigned __int8 *)(v45 - 24);
              v44 = (unsigned int)(v46 - 30);
              if ( (unsigned int)v44 > 0xA )
                goto LABEL_85;
              if ( (_BYTE)v46 == 31 )
              {
                v44 = *(_DWORD *)(v45 - 20) & 0x7FFFFFF;
                if ( (_DWORD)v44 != 3 )
                  goto LABEL_85;
                v47 = *(_QWORD *)(v45 - 120);
                if ( !v47 )
                  goto LABEL_85;
              }
              else
              {
                if ( (_BYTE)v46 != 32 )
                  goto LABEL_85;
                v47 = **(_QWORD **)(v45 - 32);
                if ( !v47 )
                  goto LABEL_85;
              }
              if ( a3 == v47 )
                goto LABEL_7;
              if ( v77 || *(_BYTE *)v47 != 5 && *(_BYTE *)v47 <= 0x1Cu )
                goto LABEL_85;
              v48 = 32LL * (*(_DWORD *)(v47 + 4) & 0x7FFFFFF);
              if ( (*(_BYTE *)(v47 + 7) & 0x40) != 0 )
              {
                v49 = *(char **)(v47 - 8);
                v73 = &v49[v48];
              }
              else
              {
                v73 = (char *)v47;
                v49 = (char *)(v47 - v48);
              }
              v50 = v48 >> 7;
              if ( v50 )
              {
                v51 = &v49[128 * v50];
                while ( a3 != *(_QWORD *)v49 || !(unsigned __int8)sub_98D0D0((__int64)v49, v31, v44, v35, v32) )
                {
                  if ( a3 == *((_QWORD *)v49 + 4)
                    && (v52 = v49 + 32, (unsigned __int8)sub_98D0D0((__int64)(v49 + 32), v31, v44, v35, v32))
                    || a3 == *((_QWORD *)v49 + 8)
                    && (v52 = v49 + 64, (unsigned __int8)sub_98D0D0((__int64)(v49 + 64), v31, v44, v35, v32))
                    || a3 == *((_QWORD *)v49 + 12)
                    && (v52 = v49 + 96, (unsigned __int8)sub_98D0D0((__int64)(v49 + 96), v31, v44, v35, v32)) )
                  {
                    v49 = v52;
                    break;
                  }
                  v49 += 128;
                  if ( v51 == v49 )
                    goto LABEL_128;
                }
LABEL_106:
                if ( v73 != v49 )
                  goto LABEL_7;
                goto LABEL_85;
              }
LABEL_128:
              v59 = v73 - v49;
              if ( v73 - v49 != 64 )
              {
                if ( v59 != 96 )
                {
                  if ( v59 != 32 )
                    goto LABEL_85;
                  goto LABEL_131;
                }
                if ( a3 == *(_QWORD *)v49 && (unsigned __int8)sub_98D0D0((__int64)v49, v31, v44, v35, v32) )
                  goto LABEL_106;
                v49 += 32;
              }
              if ( a3 == *(_QWORD *)v49 && (unsigned __int8)sub_98D0D0((__int64)v49, v31, v44, v35, v32) )
                goto LABEL_106;
              v49 += 32;
LABEL_131:
              if ( a3 == *(_QWORD *)v49 && (unsigned __int8)sub_98D0D0((__int64)v49, v31, v44, v35, v32) )
                goto LABEL_106;
LABEL_85:
              v43 = (_QWORD *)v43[1];
              if ( !v43 )
              {
                LODWORD(v9) = a3;
                LODWORD(a4) = v69;
                LODWORD(a3) = v71;
                goto LABEL_53;
              }
            }
          }
        }
      }
    }
  }
  return v15;
}
