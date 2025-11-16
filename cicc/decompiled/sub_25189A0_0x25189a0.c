// Function: sub_25189A0
// Address: 0x25189a0
//
__int64 __fastcall sub_25189A0(__int64 *a1, unsigned __int64 a2, unsigned __int8 *a3, unsigned __int64 a4, __int64 a5)
{
  unsigned __int8 *v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // r9
  char v10; // di
  __int64 v11; // r10
  int v12; // edx
  __int64 v13; // rax
  unsigned __int8 *v14; // r11
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  _BYTE *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r15
  int v23; // edx
  __int64 result; // rax
  unsigned __int8 *v25; // r15
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // r14
  int v30; // r14d
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rbx
  __int64 *v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rdi
  _BYTE *v37; // rbx
  int v38; // eax
  __int64 *v39; // r9
  __int64 v40; // rax
  bool v41; // r15
  _QWORD *v42; // rdi
  _QWORD *v43; // rsi
  unsigned __int8 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r15
  __int64 v48; // r14
  __int64 v49; // rdi
  int v50; // edx
  __int64 v51; // rsi
  int v52; // edx
  unsigned int v53; // eax
  unsigned __int8 *v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // r12
  __int64 v57; // r12
  int v58; // r10d
  char v59; // al
  __int64 v60; // r8
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 *v63; // rbx
  unsigned __int64 v64; // rsi
  __int64 v65; // r9
  unsigned __int64 v66; // rdx
  unsigned __int64 *v67; // rdi
  __int64 v68; // rdx
  int v69; // r14d
  __int64 *v70; // r9
  __int64 v71; // r14
  _QWORD *v72; // rdi
  __int64 v73; // r12
  int v74; // esi
  __int64 v75; // r10
  unsigned int v76; // eax
  unsigned __int8 *v77; // rcx
  char *v78; // rbx
  int v79; // edx
  __int64 v80; // [rsp+8h] [rbp-88h]
  __int64 *v81; // [rsp+10h] [rbp-80h]
  __int64 v82; // [rsp+10h] [rbp-80h]
  __int64 v83; // [rsp+10h] [rbp-80h]
  __int64 v84; // [rsp+10h] [rbp-80h]
  __int64 v85; // [rsp+10h] [rbp-80h]
  __int64 v86; // [rsp+10h] [rbp-80h]
  __int64 v88; // [rsp+20h] [rbp-70h] BYREF
  __int64 v89; // [rsp+28h] [rbp-68h]
  unsigned __int8 *v90; // [rsp+30h] [rbp-60h]
  __int64 v91; // [rsp+40h] [rbp-50h] BYREF
  __int64 v92; // [rsp+48h] [rbp-48h]
  __int64 v93; // [rsp+50h] [rbp-40h]

  v7 = *(unsigned __int8 **)a2;
  v8 = *a1;
  v9 = *a1 + 1648;
  v10 = *(_BYTE *)(*a1 + 1640) & 1;
  while ( 1 )
  {
    if ( v10 )
    {
      v11 = v8 + 1648;
      v12 = 31;
    }
    else
    {
      v17 = *(unsigned int *)(v8 + 1656);
      v11 = *(_QWORD *)(v8 + 1648);
      if ( !(_DWORD)v17 )
        goto LABEL_13;
      v12 = v17 - 1;
    }
    a4 = v12 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v13 = v11 + 16 * a4;
    v14 = *(unsigned __int8 **)v13;
    if ( *(unsigned __int8 **)v13 == a3 )
      goto LABEL_4;
    v38 = 1;
    while ( v14 != (unsigned __int8 *)-4096LL )
    {
      v69 = v38 + 1;
      a4 = v12 & (unsigned int)(v38 + a4);
      v13 = v11 + 16LL * (unsigned int)a4;
      v14 = *(unsigned __int8 **)v13;
      if ( *(unsigned __int8 **)v13 == a3 )
        goto LABEL_4;
      v38 = v69;
    }
    if ( v10 )
    {
      v18 = 512;
      goto LABEL_14;
    }
    v17 = *(unsigned int *)(v8 + 1656);
LABEL_13:
    v18 = 16 * v17;
LABEL_14:
    v13 = v11 + v18;
LABEL_4:
    v15 = 512;
    if ( !v10 )
      v15 = 16LL * *(unsigned int *)(v8 + 1656);
    if ( v13 == v11 + v15 )
      break;
    v16 = *(_QWORD *)(*(_QWORD *)(v8 + 2160) + 16LL * *(unsigned int *)(v13 + 8) + 8) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v16 )
      break;
    a3 = (unsigned __int8 *)v16;
  }
  v19 = *(_BYTE **)(a2 + 24);
  if ( *v19 != 30 )
  {
    if ( v7 )
    {
LABEL_17:
      v20 = *(_QWORD *)(a2 + 8);
      **(_QWORD **)(a2 + 16) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = *(_QWORD *)(a2 + 16);
    }
    *(_QWORD *)a2 = a3;
    if ( a3 )
      goto LABEL_20;
    goto LABEL_23;
  }
  v44 = sub_BD3990(v7, v8);
  if ( *v44 != 85 || (*((_WORD *)v44 + 1) & 3) != 2 )
    goto LABEL_56;
  v88 = 4;
  v70 = &v88;
  v89 = 0;
  v90 = v44;
  v71 = *a1;
  if ( v44 != (unsigned __int8 *)-4096LL && v44 != (unsigned __int8 *)-8192LL )
  {
    sub_BD73F0((__int64)&v88);
    v70 = &v88;
  }
  if ( *(_DWORD *)(v71 + 3896) )
  {
    v74 = *(_DWORD *)(v71 + 3904);
    if ( v74 )
    {
      v8 = (unsigned int)(v74 - 1);
      v75 = *(_QWORD *)(v71 + 3888);
      v91 = 4;
      v92 = 0;
      v93 = -4096;
      v76 = v8 & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
      v77 = *(unsigned __int8 **)(v75 + 24LL * v76 + 16);
      if ( v90 == v77 )
      {
LABEL_106:
        sub_D68D70(&v91);
        v70 = &v88;
        goto LABEL_107;
      }
      v79 = 1;
      while ( v77 != (unsigned __int8 *)-4096LL )
      {
        v76 = v8 & (v79 + v76);
        v77 = *(unsigned __int8 **)(v75 + 24LL * v76 + 16);
        if ( v90 == v77 )
          goto LABEL_106;
        ++v79;
      }
      sub_D68D70(&v91);
      v70 = &v88;
    }
    return sub_D68D70(v70);
  }
  v72 = *(_QWORD **)(v71 + 3912);
  v8 = (__int64)&v72[3 * *(unsigned int *)(v71 + 3920)];
  if ( (_QWORD *)v8 == sub_2506680(v72, v8, (__int64)&v88) )
    return sub_D68D70(v70);
LABEL_107:
  sub_D68D70(v70);
LABEL_56:
  if ( *a3 == 22 )
    goto LABEL_125;
  v45 = sub_B43CB0((__int64)v19);
  if ( (*(_BYTE *)(v45 + 2) & 1) != 0 )
  {
    v84 = v45;
    sub_B2C6D0(v45, v8, v46, a4);
    v47 = *(_QWORD *)(v84 + 96);
    v48 = v47 + 40LL * *(_QWORD *)(v84 + 104);
    if ( (*(_BYTE *)(v84 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v84, v8, 5LL * *(_QWORD *)(v84 + 104), a4);
      v47 = *(_QWORD *)(v84 + 96);
    }
  }
  else
  {
    v47 = *(_QWORD *)(v45 + 96);
    v48 = v47 + 40LL * *(_QWORD *)(v45 + 104);
  }
  if ( v47 == v48 )
  {
LABEL_125:
    if ( *(_QWORD *)a2 )
      goto LABEL_17;
  }
  else
  {
    do
    {
      v49 = v47;
      v47 += 40;
      sub_B2D5C0(v49, 52);
    }
    while ( v48 != v47 );
    if ( *(_QWORD *)a2 )
      goto LABEL_17;
  }
  *(_QWORD *)a2 = a3;
LABEL_20:
  v21 = *((_QWORD *)a3 + 2);
  *(_QWORD *)(a2 + 8) = v21;
  if ( v21 )
  {
    a4 = a2 + 8;
    *(_QWORD *)(v21 + 16) = a2 + 8;
  }
  *(_QWORD *)(a2 + 16) = a3 + 16;
  *((_QWORD *)a3 + 2) = a2;
LABEL_23:
  if ( *v7 > 0x1Cu )
  {
    v22 = *a1 + 288;
    v91 = sub_B43CB0((__int64)v7);
    sub_2518560(v22, &v91);
    if ( *v7 != 84 )
    {
      v90 = v7;
      v39 = &v88;
      v88 = 4;
      v89 = 0;
      v40 = *a1;
      v41 = v7 + 4096 != 0 && v7 + 0x2000 != 0;
      if ( v41 )
      {
        v80 = *a1;
        sub_BD73F0((__int64)&v88);
        v40 = v80;
        v39 = &v88;
        if ( !*(_DWORD *)(v80 + 3896) )
        {
LABEL_53:
          v42 = *(_QWORD **)(v40 + 3912);
          v43 = &v42[3 * *(unsigned int *)(v40 + 3920)];
          if ( v43 != sub_2506680(v42, (__int64)v43, (__int64)&v88) )
            goto LABEL_54;
          goto LABEL_77;
        }
      }
      else if ( !*(_DWORD *)(v40 + 3896) )
      {
        goto LABEL_53;
      }
      v50 = *(_DWORD *)(v40 + 3904);
      if ( v50 )
      {
        v51 = *(_QWORD *)(v40 + 3888);
        v52 = v50 - 1;
        v91 = 4;
        v92 = 0;
        v93 = -4096;
        v53 = v52 & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
        v54 = *(unsigned __int8 **)(v51 + 24LL * v53 + 16);
        if ( v54 == v90 )
        {
LABEL_68:
          sub_D68D70(&v91);
          v39 = &v88;
          goto LABEL_54;
        }
        v58 = 1;
        while ( v54 != (unsigned __int8 *)-4096LL )
        {
          v53 = v52 & (v58 + v53);
          v54 = *(unsigned __int8 **)(v51 + 24LL * v53 + 16);
          if ( v90 == v54 )
            goto LABEL_68;
          ++v58;
        }
        sub_D68D70(&v91);
        v39 = &v88;
      }
LABEL_77:
      v81 = v39;
      v59 = sub_F50EE0(v7, 0);
      v39 = v81;
      if ( v59 )
      {
        sub_D68D70(v81);
        v93 = (__int64)v7;
        v91 = 6;
        v92 = 0;
        v61 = a1[1];
        if ( v41 )
        {
          v82 = a1[1];
          sub_BD73F0((__int64)&v91);
          v61 = v82;
        }
        v62 = *(unsigned int *)(v61 + 8);
        v63 = &v91;
        v64 = *(_QWORD *)v61;
        v65 = v62 + 1;
        v66 = v62;
        if ( v62 + 1 > (unsigned __int64)*(unsigned int *)(v61 + 12) )
        {
          if ( v64 > (unsigned __int64)&v91 || (v66 = v64 + 24 * v62, (unsigned __int64)&v91 >= v66) )
          {
            v86 = v61;
            v63 = &v91;
            sub_F39130(v61, v62 + 1, v66, v62, v60, v65);
            v61 = v86;
            v62 = *(unsigned int *)(v86 + 8);
            v64 = *(_QWORD *)v86;
            LODWORD(v66) = *(_DWORD *)(v86 + 8);
          }
          else
          {
            v78 = (char *)&v91 - v64;
            v85 = v61;
            sub_F39130(v61, v62 + 1, v66, v62, v60, v65);
            v61 = v85;
            v64 = *(_QWORD *)v85;
            v62 = *(unsigned int *)(v85 + 8);
            v63 = (__int64 *)&v78[*(_QWORD *)v85];
            LODWORD(v66) = *(_DWORD *)(v85 + 8);
          }
        }
        v67 = (unsigned __int64 *)(v64 + 24 * v62);
        if ( v67 )
        {
          *v67 = 6;
          v68 = v63[2];
          v67[1] = 0;
          v67[2] = v68;
          if ( v68 != -4096 && v68 != 0 && v68 != -8192 )
          {
            v83 = v61;
            sub_BD6050(v67, *v63 & 0xFFFFFFFFFFFFFFF8LL);
            v61 = v83;
          }
          LODWORD(v66) = *(_DWORD *)(v61 + 8);
        }
        *(_DWORD *)(v61 + 8) = v66 + 1;
        sub_D68D70(&v91);
        goto LABEL_25;
      }
LABEL_54:
      sub_D68D70(v39);
    }
  }
LABEL_25:
  v23 = *a3;
  result = (unsigned int)(v23 - 12);
  if ( (unsigned int)result > 1 )
    goto LABEL_44;
  v25 = *(unsigned __int8 **)(a2 + 24);
  result = *v25;
  if ( (unsigned __int8)result <= 0x1Cu )
    goto LABEL_44;
  a4 = (unsigned int)(result - 34);
  if ( (unsigned __int8)(result - 34) > 0x33u )
    goto LABEL_44;
  v26 = 0x8000000000041LL;
  if ( _bittest64(&v26, a4) )
  {
    a4 = (unsigned __int64)&v25[-32 * (*((_DWORD *)v25 + 1) & 0x7FFFFFF)];
    if ( a2 >= a4 )
    {
      if ( (_DWORD)result == 40 )
      {
        result = 32LL * (unsigned int)sub_B491D0(*(_QWORD *)(a2 + 24));
        v27 = -32 - result;
      }
      else
      {
        v27 = -32;
        if ( (_DWORD)result != 85 )
        {
          v27 = -96;
          if ( (_DWORD)result != 34 )
            BUG();
        }
      }
      if ( (v25[7] & 0x80u) != 0 )
      {
        result = sub_BD2BC0((__int64)v25);
        v29 = result + v28;
        if ( (v25[7] & 0x80u) == 0 )
        {
          if ( !(unsigned int)(v29 >> 4) )
            goto LABEL_38;
        }
        else
        {
          result = sub_BD2BC0((__int64)v25);
          if ( !(unsigned int)((v29 - result) >> 4) )
            goto LABEL_38;
          if ( (v25[7] & 0x80u) != 0 )
          {
            v30 = *(_DWORD *)(sub_BD2BC0((__int64)v25) + 8);
            if ( (v25[7] & 0x80u) == 0 )
              BUG();
            v31 = sub_BD2BC0((__int64)v25);
            result = 32LL * (unsigned int)(*(_DWORD *)(v31 + v32 - 4) - v30);
            v27 -= result;
            goto LABEL_38;
          }
        }
        BUG();
      }
LABEL_38:
      if ( a2 < (unsigned __int64)&v25[v27] )
      {
        v33 = a2 - (_QWORD)&v25[-32 * (*((_DWORD *)v25 + 1) & 0x7FFFFFF)];
        v34 = (__int64 *)sub_BD5C60((__int64)v25);
        v35 = v33 >> 5;
        result = sub_A7B980((__int64 *)v25 + 9, v34, (int)v35 + 1, 40);
        v36 = *((_QWORD *)v25 - 4);
        *((_QWORD *)v25 + 9) = result;
        if ( v36 )
        {
          if ( !*(_BYTE *)v36 )
          {
            result = (unsigned int)v35;
            if ( (unsigned __int64)(unsigned int)v35 < *(_QWORD *)(v36 + 104) )
              result = sub_B2D580(v36, v35, 40);
          }
        }
      }
      v23 = *a3;
LABEL_44:
      if ( (unsigned __int8)v23 <= 0x15u )
      {
        v37 = *(_BYTE **)(a2 + 24);
        if ( *v37 == 31 )
        {
          v55 = (unsigned int)(v23 - 12);
          if ( (unsigned __int8)v55 > 1u )
          {
            v73 = a1[2];
            result = *(unsigned int *)(v73 + 8);
            if ( result + 1 > (unsigned __int64)*(unsigned int *)(v73 + 12) )
            {
              sub_C8D5F0(v73, (const void *)(v73 + 16), result + 1, 8u, a5, v9);
              result = *(unsigned int *)(v73 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v73 + 8 * result) = v37;
            ++*(_DWORD *)(v73 + 8);
          }
          else
          {
            v56 = *a1;
            v91 = 4;
            v92 = 0;
            v93 = (__int64)v37;
            v57 = v56 + 2688;
            if ( v37 != (_BYTE *)-8192LL && v37 != (_BYTE *)-4096LL )
              sub_BD73F0((__int64)&v91);
            sub_2518170(v57, (__int64)&v91, v55, a4, a5, v9);
            return sub_D68D70(&v91);
          }
        }
      }
    }
  }
  return result;
}
