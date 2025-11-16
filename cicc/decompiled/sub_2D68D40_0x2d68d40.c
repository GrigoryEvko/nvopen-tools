// Function: sub_2D68D40
// Address: 0x2d68d40
//
__int64 __fastcall sub_2D68D40(__int64 a1, unsigned __int64 a2)
{
  bool v4; // r13
  int v5; // eax
  __int64 v6; // rcx
  int v7; // esi
  unsigned int v8; // eax
  __int64 v9; // r14
  __int64 v10; // rdi
  _QWORD *v11; // rsi
  char *v12; // rsi
  __int64 v13; // rcx
  char *v14; // rdx
  __int64 result; // rax
  char *v16; // rdi
  char *v17; // r14
  __int64 v18; // rdx
  unsigned __int64 v19; // r13
  __int64 v20; // rax
  int *v21; // r14
  int *v22; // r13
  int *v23; // r15
  int *v24; // rcx
  bool v25; // al
  int v26; // edx
  int v27; // edx
  __int64 v28; // rsi
  unsigned int v29; // eax
  _QWORD *v30; // r13
  __int64 v31; // rdi
  int v32; // eax
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rsi
  unsigned int v36; // ecx
  __int64 v37; // rdx
  __int64 v38; // r10
  __int64 v39; // r13
  __int64 v40; // rcx
  __int64 v41; // rdx
  char *v42; // r13
  __int64 v43; // rsi
  __int64 v44; // rdx
  __int64 v45; // rbx
  _QWORD *v46; // rdx
  char *v47; // r14
  __int64 v48; // r15
  __int64 v49; // rcx
  __int64 v50; // r12
  _QWORD *v51; // r14
  __int64 v52; // r15
  __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // rdx
  int *v56; // rsi
  __int64 v57; // rsi
  int *v58; // r8
  __int64 v59; // rax
  int v60; // r8d
  int v61; // edx
  int v62; // r11d
  int v63; // r8d
  __int64 v64; // [rsp+0h] [rbp-90h]
  bool v65; // [rsp+Fh] [rbp-81h]
  bool v66; // [rsp+10h] [rbp-80h]
  char *v67; // [rsp+10h] [rbp-80h]
  __int64 v68; // [rsp+18h] [rbp-78h]
  int *v69; // [rsp+18h] [rbp-78h]
  int *v70; // [rsp+18h] [rbp-78h]
  __int64 v71; // [rsp+20h] [rbp-70h] BYREF
  __int64 v72; // [rsp+28h] [rbp-68h]
  unsigned __int64 v73; // [rsp+30h] [rbp-60h]
  __int64 v74; // [rsp+40h] [rbp-50h] BYREF
  __int64 v75; // [rsp+48h] [rbp-48h]
  unsigned __int64 v76; // [rsp+50h] [rbp-40h]

  v73 = a2;
  v65 = a2 != -4096;
  v66 = a2 != -8192;
  v64 = a1 + 568;
  v71 = 0;
  v72 = 0;
  v4 = v66 && v65 && a2 != 0;
  if ( v4 )
    sub_BD73F0((__int64)&v71);
  v5 = *(_DWORD *)(a1 + 592);
  v6 = *(_QWORD *)(a1 + 576);
  if ( v5 )
  {
    v7 = v5 - 1;
    v74 = 0;
    v75 = 0;
    v76 = -4096;
    v8 = (v5 - 1) & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
    v9 = v6 + 32LL * v8;
    v10 = *(_QWORD *)(v9 + 16);
    if ( v73 == v10 )
    {
LABEL_5:
      sub_D68D70(&v74);
      if ( v9 != *(_QWORD *)(a1 + 576) + 32LL * *(unsigned int *)(a1 + 592) )
      {
        v11 = (_QWORD *)(*(_QWORD *)(a1 + 600) + 1064LL * *(unsigned int *)(v9 + 24));
        if ( v11 != (_QWORD *)(*(_QWORD *)(a1 + 600) + 1064LL * *(unsigned int *)(a1 + 608)) )
          sub_2D689E0(v64, v11);
      }
    }
    else
    {
      v60 = 1;
      while ( v10 != -4096 )
      {
        v8 = v7 & (v60 + v8);
        v9 = v6 + 32LL * v8;
        v10 = *(_QWORD *)(v9 + 16);
        if ( v73 == v10 )
          goto LABEL_5;
        ++v60;
      }
      sub_D68D70(&v74);
    }
  }
  if ( v73 != -4096 && v73 != 0 && v73 != -8192 )
    sub_BD60C0(&v71);
  v74 = 0;
  v75 = 0;
  v76 = a2;
  if ( v4 )
    sub_BD73F0((__int64)&v74);
  if ( *(_QWORD *)(a1 + 720) )
  {
    v21 = (int *)(a1 + 688);
    if ( *(_QWORD *)(a1 + 696) )
    {
      result = v76;
      v22 = (int *)(a1 + 688);
      v23 = *(int **)(a1 + 696);
      while ( 1 )
      {
        while ( *((_QWORD *)v23 + 6) < v76 )
        {
          v23 = (int *)*((_QWORD *)v23 + 3);
          if ( !v23 )
            goto LABEL_37;
        }
        v24 = (int *)*((_QWORD *)v23 + 2);
        if ( *((_QWORD *)v23 + 6) <= v76 )
          break;
        v22 = v23;
        v23 = (int *)*((_QWORD *)v23 + 2);
        if ( !v24 )
        {
LABEL_37:
          v25 = v21 == v22;
          goto LABEL_38;
        }
      }
      v56 = (int *)*((_QWORD *)v23 + 3);
      while ( v56 )
      {
        if ( v76 >= *((_QWORD *)v56 + 6) )
        {
          v56 = (int *)*((_QWORD *)v56 + 3);
        }
        else
        {
          v22 = v56;
          v56 = (int *)*((_QWORD *)v56 + 2);
        }
      }
      while ( v24 )
      {
        while ( 1 )
        {
          v57 = *((_QWORD *)v24 + 3);
          if ( v76 <= *((_QWORD *)v24 + 6) )
            break;
          v24 = (int *)*((_QWORD *)v24 + 3);
          if ( !v57 )
            goto LABEL_104;
        }
        v23 = v24;
        v24 = (int *)*((_QWORD *)v24 + 2);
      }
LABEL_104:
      if ( *(int **)(a1 + 704) != v23 || v21 != v22 )
      {
        if ( v23 == v22 )
          goto LABEL_26;
        do
        {
          v69 = v23;
          v23 = (int *)sub_220EF30((__int64)v23);
          v58 = sub_220F330(v69, (_QWORD *)(a1 + 688));
          v59 = *((_QWORD *)v58 + 6);
          if ( v59 != -4096 && v59 != 0 && v59 != -8192 )
          {
            v70 = v58;
            sub_BD60C0((_QWORD *)v58 + 4);
            v58 = v70;
          }
          j_j___libc_free_0((unsigned __int64)v58);
          --*(_QWORD *)(a1 + 720);
        }
        while ( v23 != v22 );
        goto LABEL_25;
      }
    }
    else
    {
      v22 = (int *)(a1 + 688);
      v25 = 1;
LABEL_38:
      if ( *(int **)(a1 + 704) != v22 || !v25 )
        goto LABEL_25;
    }
    sub_2D587C0(*(_QWORD **)(a1 + 696));
    *(_QWORD *)(a1 + 704) = v21;
    result = v76;
    *(_QWORD *)(a1 + 696) = 0;
    *(_QWORD *)(a1 + 712) = v21;
    *(_QWORD *)(a1 + 720) = 0;
    goto LABEL_26;
  }
  v12 = *(char **)(a1 + 616);
  v13 = *(unsigned int *)(a1 + 624);
  v14 = &v12[24 * v13];
  result = v76;
  if ( v12 != v14 )
  {
    v16 = *(char **)(a1 + 616);
    while ( *((_QWORD *)v16 + 2) != v76 )
    {
      v16 += 24;
      if ( v14 == v16 )
        goto LABEL_26;
    }
    if ( v14 != v16 )
    {
      v17 = v16 + 24;
      v18 = v14 - (v16 + 24);
      v19 = 0xAAAAAAAAAAAAAAABLL * (v18 >> 3);
      if ( v18 > 0 )
      {
        while ( 1 )
        {
          sub_2D57220(v16, *((_QWORD *)v16 + 5));
          v16 = v17;
          if ( !--v19 )
            break;
          v17 += 24;
        }
        LODWORD(v13) = *(_DWORD *)(a1 + 624);
        v12 = *(char **)(a1 + 616);
      }
      v20 = (unsigned int)(v13 - 1);
      *(_DWORD *)(a1 + 624) = v20;
      sub_D68D70(&v12[24 * v20]);
LABEL_25:
      result = v76;
    }
  }
LABEL_26:
  if ( result != 0 && result != -4096 && result != -8192 )
    result = sub_BD60C0(&v74);
  if ( *(_BYTE *)a2 == 63 )
  {
    v71 = 0;
    v72 = 0;
    v73 = a2;
    if ( v66 && v65 )
      sub_BD73F0((__int64)&v71);
    v26 = *(_DWORD *)(a1 + 752);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 736);
      v74 = 0;
      v75 = 0;
      v76 = -4096;
      v29 = v27 & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
      v30 = (_QWORD *)(v28 + 32LL * v29);
      v31 = v30[2];
      if ( v73 == v31 )
      {
LABEL_46:
        sub_D68D70(&v74);
        v74 = 0;
        v75 = 0;
        v76 = -8192;
        sub_2D57220(v30, -8192);
        sub_D68D70(&v74);
        --*(_DWORD *)(a1 + 744);
        ++*(_DWORD *)(a1 + 748);
      }
      else
      {
        v63 = 1;
        while ( v31 != -4096 )
        {
          v29 = v27 & (v63 + v29);
          v30 = (_QWORD *)(v28 + 32LL * v29);
          v31 = v30[2];
          if ( v73 == v31 )
            goto LABEL_46;
          ++v63;
        }
        sub_D68D70(&v74);
      }
    }
    sub_D68D70(&v71);
    v32 = *(_DWORD *)(a2 + 4);
    v74 = 0;
    v75 = 0;
    v33 = *(_QWORD *)(a2 - 32LL * (v32 & 0x7FFFFFF));
    if ( v33 )
    {
      v76 = v33;
      if ( v33 != -8192 && v33 != -4096 )
        sub_BD73F0((__int64)&v74);
    }
    else
    {
      v76 = 0;
    }
    v34 = *(unsigned int *)(a1 + 592);
    if ( (_DWORD)v34 )
    {
      v35 = *(_QWORD *)(a1 + 576);
      v36 = (v34 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
      v37 = v35 + 32LL * v36;
      v38 = *(_QWORD *)(v37 + 16);
      if ( v76 == v38 )
      {
LABEL_53:
        v39 = *(_QWORD *)(a1 + 600);
        if ( v37 != v35 + 32 * v34 )
        {
          v68 = v39 + 1064LL * *(unsigned int *)(v37 + 24);
          goto LABEL_55;
        }
LABEL_116:
        v68 = v39 + 1064LL * *(unsigned int *)(a1 + 608);
LABEL_55:
        sub_D68D70(&v74);
        result = *(_QWORD *)(a1 + 600) + 1064LL * *(unsigned int *)(a1 + 608);
        if ( v68 == result )
          return result;
        v40 = *(_QWORD *)(v68 + 24);
        v41 = 32LL * *(unsigned int *)(v68 + 32);
        v42 = (char *)(v40 + v41);
        v43 = v41 >> 5;
        v44 = v41 >> 7;
        if ( v44 )
        {
          v45 = *(_QWORD *)(v68 + 24);
          v46 = (_QWORD *)(v40 + (v44 << 7));
          while ( a2 != *(_QWORD *)(v45 + 16) )
          {
            if ( a2 == *(_QWORD *)(v45 + 48) )
            {
              v45 += 32;
              break;
            }
            if ( a2 == *(_QWORD *)(v45 + 80) )
            {
              v45 += 64;
              break;
            }
            if ( a2 == *(_QWORD *)(v45 + 112) )
            {
              v45 += 96;
              break;
            }
            v45 += 128;
            if ( v46 == (_QWORD *)v45 )
            {
              v43 = (__int64)&v42[-v45] >> 5;
              goto LABEL_122;
            }
          }
LABEL_63:
          if ( v42 != (char *)v45 )
          {
            v47 = (char *)(v45 + 32);
            if ( v42 == (char *)(v45 + 32) )
            {
              v42 = (char *)v45;
            }
            else
            {
              do
              {
                v48 = *((_QWORD *)v47 + 2);
                if ( a2 != v48 )
                {
                  v49 = *(_QWORD *)(v45 + 16);
                  if ( v48 != v49 )
                  {
                    if ( v49 != 0 && v49 != -4096 && v49 != -8192 )
                      sub_BD60C0((_QWORD *)v45);
                    *(_QWORD *)(v45 + 16) = v48;
                    if ( v48 != 0 && v48 != -4096 && v48 != -8192 )
                      sub_BD73F0(v45);
                  }
                  v45 += 32;
                  *(_QWORD *)(v45 - 8) = *((_QWORD *)v47 + 3);
                }
                v47 += 32;
              }
              while ( v42 != v47 );
              v40 = *(_QWORD *)(v68 + 24);
              v47 = (char *)(v40 + 32LL * *(unsigned int *)(v68 + 32));
              v67 = (char *)(v47 - v42);
              v50 = (v47 - v42) >> 5;
              if ( v47 - v42 <= 0 )
              {
                v42 = (char *)v45;
              }
              else
              {
                v51 = (_QWORD *)v45;
                do
                {
                  v52 = *((_QWORD *)v42 + 2);
                  v53 = v51[2];
                  if ( v52 != v53 )
                  {
                    if ( v53 != 0 && v53 != -4096 && v53 != -8192 )
                      sub_BD60C0(v51);
                    v51[2] = v52;
                    if ( v52 != 0 && v52 != -4096 && v52 != -8192 )
                      sub_BD73F0((__int64)v51);
                  }
                  v54 = *((_QWORD *)v42 + 3);
                  v51 += 4;
                  v42 += 32;
                  *(v51 - 1) = v54;
                  --v50;
                }
                while ( v50 );
                v42 = &v67[v45];
                v40 = *(_QWORD *)(v68 + 24);
                v47 = (char *)(v40 + 32LL * *(unsigned int *)(v68 + 32));
              }
              if ( v42 == v47 )
                goto LABEL_92;
            }
            do
            {
              v55 = *((_QWORD *)v47 - 2);
              v47 -= 32;
              if ( v55 != 0 && v55 != -4096 && v55 != -8192 )
                sub_BD60C0(v47);
            }
            while ( v42 != v47 );
            v40 = *(_QWORD *)(v68 + 24);
          }
LABEL_92:
          result = (__int64)&v42[-v40] >> 5;
          *(_DWORD *)(v68 + 32) = result;
          if ( !(_DWORD)result )
            return sub_2D689E0(v64, (_QWORD *)v68);
          return result;
        }
        v45 = *(_QWORD *)(v68 + 24);
LABEL_122:
        if ( v43 != 2 )
        {
          if ( v43 != 3 )
          {
            if ( v43 != 1 )
              goto LABEL_92;
            goto LABEL_125;
          }
          if ( a2 == *(_QWORD *)(v45 + 16) )
            goto LABEL_63;
          v45 += 32;
        }
        if ( a2 == *(_QWORD *)(v45 + 16) )
          goto LABEL_63;
        v45 += 32;
LABEL_125:
        if ( a2 != *(_QWORD *)(v45 + 16) )
          goto LABEL_92;
        goto LABEL_63;
      }
      v61 = 1;
      while ( v38 != -4096 )
      {
        v62 = v61 + 1;
        v36 = (v34 - 1) & (v61 + v36);
        v37 = v35 + 32LL * v36;
        v38 = *(_QWORD *)(v37 + 16);
        if ( v76 == v38 )
          goto LABEL_53;
        v61 = v62;
      }
    }
    v39 = *(_QWORD *)(a1 + 600);
    goto LABEL_116;
  }
  return result;
}
