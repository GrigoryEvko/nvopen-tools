// Function: sub_DF86E0
// Address: 0xdf86e0
//
size_t __fastcall sub_DF86E0(__int64 a1, int a2, unsigned __int8 *a3, __int64 a4, __int64 a5, char a6, __int64 a7)
{
  unsigned __int8 *v7; // rax
  __int64 v10; // rax
  const void *v11; // rsi
  int v12; // eax
  __int64 v13; // r13
  int v14; // edx
  unsigned int v15; // ecx
  unsigned __int8 v16; // al
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rax
  bool v21; // al
  __int64 *v22; // rax
  int v23; // edx
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rax
  int v29; // r13d
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // r8
  unsigned __int64 v34; // rcx
  __int64 v35; // r10
  unsigned __int8 *v36; // r13
  __int64 v37; // r10
  unsigned __int64 v38; // rax
  signed __int64 v39; // rdi
  __int64 v40; // r9
  unsigned __int64 v41; // r11
  char *v42; // r15
  __int64 v43; // rcx
  char *v44; // r14
  unsigned int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rax
  _QWORD *v48; // rdi
  __int64 v49; // r8
  signed __int64 v50; // r11
  char *v51; // rdi
  __int64 v52; // rax
  bool v53; // zf
  __int64 v54; // [rsp+8h] [rbp-68h]
  __int64 v55; // [rsp+10h] [rbp-60h]
  signed __int64 v56; // [rsp+10h] [rbp-60h]
  __int64 v57; // [rsp+18h] [rbp-58h]
  __int64 v58; // [rsp+18h] [rbp-58h]
  __int64 v59; // [rsp+18h] [rbp-58h]
  __int64 v60; // [rsp+20h] [rbp-50h]
  __int64 v61; // [rsp+20h] [rbp-50h]
  __int64 v62; // [rsp+28h] [rbp-48h]
  __int64 v63; // [rsp+28h] [rbp-48h]
  __int64 v64; // [rsp+28h] [rbp-48h]
  char srca; // [rsp+30h] [rbp-40h]
  char *src; // [rsp+30h] [rbp-40h]
  void *srcb; // [rsp+30h] [rbp-40h]
  __int64 v68; // [rsp+38h] [rbp-38h]
  __int64 v69; // [rsp+38h] [rbp-38h]
  __int64 v70; // [rsp+38h] [rbp-38h]
  __int64 v71; // [rsp+38h] [rbp-38h]

  v7 = 0;
  if ( *a3 == 85 )
  {
    v7 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
    if ( v7 )
    {
      if ( !*v7 && *((_QWORD *)v7 + 3) == *((_QWORD *)a3 + 10) )
      {
        v53 = (v7[33] & 0x20) == 0;
        v7 = 0;
        if ( !v53 )
          v7 = a3;
      }
      else
      {
        v7 = 0;
      }
    }
  }
  *(_QWORD *)a1 = v7;
  v10 = *((_QWORD *)a3 + 1);
  *(_DWORD *)(a1 + 16) = a2;
  v11 = (const void *)(a1 + 88);
  *(_QWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x400000000LL;
  *(_DWORD *)(a1 + 120) = 0;
  v12 = *a3;
  *(_QWORD *)(a1 + 128) = a4;
  *(_QWORD *)(a1 + 136) = a5;
  *(_QWORD *)(a1 + 144) = a7;
  if ( (unsigned __int8)v12 > 0x1Cu )
  {
    switch ( v12 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_7;
      case 'T':
      case 'U':
      case 'V':
        v13 = *((_QWORD *)a3 + 1);
        v14 = *(unsigned __int8 *)(v13 + 8);
        v15 = v14 - 17;
        v16 = *(_BYTE *)(v13 + 8);
        if ( (unsigned int)(v14 - 17) <= 1 )
          v16 = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
        if ( v16 <= 3u || v16 == 5 || (v16 & 0xFD) == 4 )
          goto LABEL_7;
        if ( (_BYTE)v14 == 15 )
        {
          if ( (*(_BYTE *)(v13 + 9) & 4) == 0 )
            break;
          srca = a6;
          v21 = sub_BCB420(*((_QWORD *)a3 + 1));
          v11 = (const void *)(a1 + 88);
          a6 = srca;
          if ( !v21 )
            break;
          v22 = *(__int64 **)(v13 + 16);
          v13 = *v22;
          v14 = *(unsigned __int8 *)(*v22 + 8);
          v15 = v14 - 17;
        }
        else if ( (_BYTE)v14 == 16 )
        {
          do
          {
            v13 = *(_QWORD *)(v13 + 24);
            LOBYTE(v14) = *(_BYTE *)(v13 + 8);
          }
          while ( (_BYTE)v14 == 16 );
          v15 = (unsigned __int8)v14 - 17;
        }
        if ( v15 <= 1 )
          LOBYTE(v14) = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
        if ( (unsigned __int8)v14 <= 3u || (_BYTE)v14 == 5 || (v14 & 0xFD) == 4 )
        {
LABEL_7:
          v17 = a3[1] >> 1;
          if ( v17 == 127 )
            v17 = -1;
          *(_DWORD *)(a1 + 120) = v17;
        }
        break;
      default:
        break;
    }
  }
  if ( !a6 )
  {
    v23 = *a3;
    v68 = a1 + 72;
    if ( v23 == 40 )
    {
      v24 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a3);
    }
    else
    {
      v24 = -32;
      if ( v23 != 85 )
      {
        v24 = -96;
        if ( v23 != 34 )
          BUG();
      }
    }
    if ( (a3[7] & 0x80u) != 0 )
    {
      v25 = sub_BD2BC0((__int64)a3);
      v27 = v25 + v26;
      v28 = 0;
      if ( (a3[7] & 0x80u) != 0 )
        v28 = sub_BD2BC0((__int64)a3);
      if ( (unsigned int)((v27 - v28) >> 4) )
      {
        if ( (a3[7] & 0x80u) == 0 )
          BUG();
        v29 = *(_DWORD *)(sub_BD2BC0((__int64)a3) + 8);
        if ( (a3[7] & 0x80u) == 0 )
          BUG();
        v30 = sub_BD2BC0((__int64)a3);
        v24 -= 32LL * (unsigned int)(*(_DWORD *)(v30 + v31 - 4) - v29);
      }
    }
    v32 = *(unsigned int *)(a1 + 80);
    v33 = (__int64)&a3[v24];
    v34 = *(unsigned int *)(a1 + 84);
    v35 = 32LL * (*((_DWORD *)a3 + 1) & 0x7FFFFFF);
    v36 = &a3[-v35];
    v37 = v24 + v35;
    v38 = v37 >> 5;
    v39 = 8 * v32;
    v40 = v37 >> 5;
    v41 = v32 + (v37 >> 5);
    if ( 8 * v32 )
    {
      if ( v41 > v34 )
      {
        v55 = v37 >> 5;
        v57 = v37;
        v62 = v37 >> 5;
        sub_C8D5F0(v68, v11, v32 + (v37 >> 5), 8u, v33, v40);
        v32 = *(unsigned int *)(a1 + 80);
        v38 = v55;
        v37 = v57;
        v33 = (__int64)&a3[v24];
        v40 = v62;
        v39 = 8 * v32;
      }
      v42 = *(char **)(a1 + 72);
      v43 = v39 >> 3;
      v44 = &v42[v39];
      if ( v39 >> 3 >= v38 )
      {
        v49 = 8 * (v32 - v38);
        src = &v42[v49];
        v50 = v39 - v49;
        v51 = &v42[v39];
        v63 = v50 >> 3;
        if ( v32 + (v50 >> 3) > (unsigned __int64)*(unsigned int *)(a1 + 84) )
        {
          v54 = v37;
          v56 = v50;
          v59 = v40;
          v61 = 8 * (v32 - v38);
          sub_C8D5F0(v68, v11, v32 + (v50 >> 3), 8u, v49, v40);
          v32 = *(unsigned int *)(a1 + 80);
          v37 = v54;
          v50 = v56;
          v40 = v59;
          v49 = v61;
          v51 = (char *)(*(_QWORD *)(a1 + 72) + 8 * v32);
        }
        if ( v44 != src )
        {
          v58 = v37;
          v60 = v40;
          v70 = v49;
          memmove(v51, src, v50);
          LODWORD(v32) = *(_DWORD *)(a1 + 80);
          v37 = v58;
          v40 = v60;
          v49 = v70;
        }
        *(_DWORD *)(a1 + 80) = v63 + v32;
        if ( v42 != src )
        {
          srcb = (void *)v37;
          v71 = v40;
          memmove(&v44[-v49], v42, v49);
          v37 = (__int64)srcb;
          v40 = v71;
        }
        if ( v37 > 0 )
        {
          v52 = 0;
          do
          {
            *(_QWORD *)&v42[v52] = *(_QWORD *)&v36[4 * v52];
            v52 += 8;
            --v40;
          }
          while ( v40 );
        }
      }
      else
      {
        v45 = v32 + v38;
        *(_DWORD *)(a1 + 80) = v45;
        if ( v42 != v44 )
        {
          v69 = v33;
          memcpy(&v42[8 * v45 - v39], v42, v39);
          v43 = v39 >> 3;
          v33 = v69;
        }
        v46 = v43;
        v47 = 0;
        if ( v43 )
        {
          do
          {
            *(_QWORD *)&v42[v47] = *(_QWORD *)&v36[4 * v47];
            v47 += 8;
            --v46;
          }
          while ( v46 );
          v36 += 32 * v43;
        }
        if ( (unsigned __int8 *)v33 != v36 )
        {
          do
          {
            if ( v44 )
              *(_QWORD *)v44 = *(_QWORD *)v36;
            v36 += 32;
            v44 += 8;
          }
          while ( v36 != (unsigned __int8 *)v33 );
        }
      }
    }
    else
    {
      if ( v41 > v34 )
      {
        v64 = v37 >> 5;
        sub_C8D5F0(v68, v11, v32 + (v37 >> 5), 8u, v33, v40);
        v32 = *(unsigned int *)(a1 + 80);
        LODWORD(v38) = v64;
        v33 = (__int64)&a3[v24];
        v39 = 8 * v32;
      }
      v48 = (_QWORD *)(*(_QWORD *)(a1 + 72) + v39);
      if ( v36 != (unsigned __int8 *)v33 )
      {
        do
        {
          if ( v48 )
            *v48 = *(_QWORD *)v36;
          v36 += 32;
          ++v48;
        }
        while ( (unsigned __int8 *)v33 != v36 );
        LODWORD(v32) = *(_DWORD *)(a1 + 80);
      }
      *(_DWORD *)(a1 + 80) = v32 + v38;
    }
  }
  v18 = *((_QWORD *)a3 - 4);
  if ( !v18 || *(_BYTE *)v18 || (v19 = *(_QWORD *)(v18 + 24), v19 != *((_QWORD *)a3 + 10)) )
    BUG();
  return sub_DF6530(
           a1 + 24,
           *(char **)(a1 + 24),
           (char *)(*(_QWORD *)(v19 + 16) + 8LL),
           (char *)(*(_QWORD *)(v19 + 16) + 8LL * *(unsigned int *)(v19 + 12)));
}
