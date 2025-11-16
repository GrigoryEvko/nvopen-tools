// Function: sub_2905260
// Address: 0x2905260
//
void __fastcall sub_2905260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  unsigned int v9; // eax
  unsigned int v10; // eax
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // r8d
  __int64 *v18; // rsi
  __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r15
  int v26; // eax
  __int64 v27; // rdi
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 *v31; // rsi
  __int64 v32; // rax
  unsigned int v33; // r12d
  __int64 *v34; // rsi
  __int64 v35; // r10
  __int64 v36; // rcx
  __int64 v37; // rbx
  unsigned __int64 v38; // r13
  __int64 v39; // rdx
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // r13
  __int64 v45; // r15
  int v46; // eax
  __int64 v47; // rsi
  int v48; // eax
  __int64 *v49; // rdx
  __int64 v50; // rdi
  __int64 v51; // rax
  _QWORD *v52; // rdi
  __int64 v53; // rsi
  _QWORD *v54; // rax
  int v55; // r8d
  __int64 v56; // rbx
  __int64 i; // r12
  __int64 v58; // rax
  __int64 v59; // rdi
  __int64 v60; // r14
  __int64 v61; // rbx
  __int64 v62; // r15
  __int64 *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // r15
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rax
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  int v75; // esi
  __int64 v76; // rax
  __int64 v77; // rax
  unsigned __int64 v78; // rsi
  int v79; // esi
  int v80; // r11d
  __int64 *v82; // [rsp+10h] [rbp-180h]
  __int64 v83; // [rsp+18h] [rbp-178h]
  __int64 v84; // [rsp+18h] [rbp-178h]
  __int64 *v85; // [rsp+20h] [rbp-170h]
  __int64 v86; // [rsp+20h] [rbp-170h]
  unsigned __int8 *v87; // [rsp+20h] [rbp-170h]
  int v88; // [rsp+20h] [rbp-170h]
  __int64 v89; // [rsp+28h] [rbp-168h]
  __int64 *v91; // [rsp+38h] [rbp-158h]
  unsigned __int8 *v92; // [rsp+40h] [rbp-150h] BYREF
  __int64 v93; // [rsp+48h] [rbp-148h] BYREF
  __int64 *v94; // [rsp+50h] [rbp-140h] BYREF
  __int64 v95; // [rsp+58h] [rbp-138h]
  _BYTE v96[304]; // [rsp+60h] [rbp-130h] BYREF

  if ( !(_BYTE)qword_5004E08 )
    return;
  v7 = *(_QWORD *)(a1 + 32);
  v94 = (__int64 *)v96;
  v95 = 0x2000000000LL;
  v83 = v7 + 72LL * *(unsigned int *)(a1 + 40);
  if ( v7 == v83 )
    return;
  v89 = a2 + 112 * a3;
  do
  {
    while ( 1 )
    {
      v9 = *(_DWORD *)(v7 + 64);
      if ( v9 )
        v10 = v9 >> 31;
      else
        LOBYTE(v10) = (__int64)(unsigned int)qword_50050A8 > *(_QWORD *)(v7 + 56);
      if ( (_BYTE)v10 )
      {
        v11 = *(_QWORD *)v7;
        v12 = *(_QWORD *)(*(_QWORD *)v7 + 16LL);
        if ( v12 )
        {
          if ( *(_QWORD *)(v12 + 8) )
            goto LABEL_9;
          v76 = sub_BD3750(*(_QWORD *)v7);
          if ( *(_BYTE *)v76 <= 0x1Cu || *(_QWORD *)(v76 + 40) != *(_QWORD *)(v11 + 40) )
          {
            v12 = *(_QWORD *)(v11 + 16);
            if ( v12 )
            {
LABEL_9:
              while ( **(_BYTE **)(v12 + 24) != 84 )
              {
                v12 = *(_QWORD *)(v12 + 8);
                if ( !v12 )
                {
                  if ( v89 == a2 )
                    goto LABEL_80;
                  goto LABEL_12;
                }
              }
              goto LABEL_21;
            }
            if ( v89 == a2 )
            {
LABEL_80:
              v13 = 0;
            }
            else
            {
LABEL_12:
              v13 = 0;
              v14 = a2;
              do
              {
                v15 = *(unsigned int *)(v14 + 24);
                v16 = *(_QWORD *)(v14 + 8);
                if ( (_DWORD)v15 )
                {
                  v17 = (v15 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
                  v18 = (__int64 *)(v16 + 8LL * v17);
                  v19 = *v18;
                  if ( v11 == *v18 )
                  {
LABEL_15:
                    v13 += v18 != (__int64 *)(v16 + 8 * v15);
                  }
                  else
                  {
                    v75 = 1;
                    while ( v19 != -4096 )
                    {
                      v17 = (v15 - 1) & (v75 + v17);
                      v88 = v75 + 1;
                      v18 = (__int64 *)(v16 + 8LL * v17);
                      v19 = *v18;
                      if ( v11 == *v18 )
                        goto LABEL_15;
                      v75 = v88;
                    }
                  }
                }
                v14 += 112;
              }
              while ( v89 != v14 );
            }
            v20 = sub_BD3960(v11);
            if ( v20 <= v13 )
              break;
          }
        }
      }
LABEL_21:
      v7 += 72;
      if ( v83 == v7 )
        goto LABEL_22;
    }
    if ( v20 == v13 )
    {
      v21 = *(unsigned int *)(v7 + 64);
      if ( (_DWORD)v21 )
      {
        if ( (int)v21 > 0 )
          goto LABEL_21;
      }
      else if ( *(__int64 *)(v7 + 56) > 0 )
      {
        goto LABEL_21;
      }
    }
    if ( *(_DWORD *)(v7 + 16) > 1u )
    {
      *(_DWORD *)(v7 + 16) = 0;
      sub_28FF860(v7 + 8, (unsigned __int8 *)v11, v21, v22, a5, a6);
    }
    v58 = *(_QWORD *)(v11 + 16);
    if ( v58 )
    {
      v59 = v6;
      v60 = v7;
      v61 = v59;
      do
      {
        v62 = *(_QWORD *)(v58 + 24);
        LOWORD(v61) = 0;
        v93 = v11;
        v63 = (__int64 *)sub_1152A40(a4, &v93, v21, v22, a5, a6);
        v87 = sub_28FEA70(*(_QWORD *)(v60 + 8), *(unsigned int *)(v60 + 16), v62 + 24, v61, *(_QWORD *)(v60 + 48), *v63);
        sub_BD2ED0(v62, v11, (__int64)v87);
        v93 = v11;
        v68 = *(_QWORD *)sub_1152A40(a4, &v93, v64, v65, v66, v67);
        v92 = v87;
        *(_QWORD *)sub_1152A40(a4, (__int64 *)&v92, (__int64)v87, v69, v70, v71) = v68;
        v58 = *(_QWORD *)(v11 + 16);
      }
      while ( v58 );
      v72 = v61;
      v7 = v60;
      v6 = v72;
    }
    v73 = (unsigned int)v95;
    v74 = (unsigned int)v95 + 1LL;
    if ( v74 > HIDWORD(v95) )
    {
      sub_C8D5F0((__int64)&v94, v96, v74, 8u, a5, a6);
      v73 = (unsigned int)v95;
    }
    v7 += 72;
    v94[v73] = v11;
    LODWORD(v95) = v95 + 1;
  }
  while ( v83 != v7 );
LABEL_22:
  v23 = (unsigned int)v95;
  v24 = (__int64)v94;
  v25 = a1;
  v26 = v95;
  v82 = &v94[(unsigned int)v95];
  if ( v82 != v94 )
  {
    v91 = v94;
    do
    {
      v27 = *(_QWORD *)(v25 + 8);
      v28 = *v91;
      v29 = *(unsigned int *)(v25 + 24);
      if ( (_DWORD)v29 )
      {
        a5 = (unsigned int)(v29 - 1);
        v23 = (unsigned int)a5 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v24 = v27 + 16 * v23;
        v30 = *(_QWORD *)v24;
        if ( v28 == *(_QWORD *)v24 )
        {
LABEL_26:
          if ( v24 != v27 + 16 * v29 )
          {
            v23 = *(_QWORD *)(v25 + 32);
            v31 = (__int64 *)(v23 + 72LL * *(unsigned int *)(v24 + 8));
            v32 = *(unsigned int *)(v25 + 40);
            v24 = v23 + 72 * v32;
            v85 = v31;
            if ( (__int64 *)v24 != v31 )
            {
              a6 = *v31;
              v33 = a5 & (((unsigned int)*v31 >> 9) ^ ((unsigned int)*v31 >> 4));
              v34 = (__int64 *)(v27 + 16LL * v33);
              v35 = *v34;
              if ( a6 == *v34 )
              {
LABEL_29:
                *v34 = -8192;
                v36 = *(unsigned int *)(v25 + 40);
                --*(_DWORD *)(v25 + 16);
                v23 = *(_QWORD *)(v25 + 32);
                ++*(_DWORD *)(v25 + 20);
                LODWORD(v32) = v36;
                v24 = v23 + 72 * v36;
              }
              else
              {
                v79 = 1;
                while ( v35 != -4096 )
                {
                  v80 = v79 + 1;
                  v33 = a5 & (v79 + v33);
                  v34 = (__int64 *)(v27 + 16LL * v33);
                  v35 = *v34;
                  if ( a6 == *v34 )
                    goto LABEL_29;
                  v79 = v80;
                }
              }
              v24 -= (__int64)(v85 + 9);
              if ( v24 > 0 )
              {
                v84 = v28;
                v37 = (__int64)(v85 + 1);
                v38 = 0x8E38E38E38E38E39LL * (v24 >> 3);
                do
                {
                  v39 = *(_QWORD *)(v37 + 64);
                  v40 = v37;
                  v37 += 72;
                  *(_QWORD *)(v37 - 80) = v39;
                  sub_28FEE40(v40, (char **)v37, v39, v24, a5, a6);
                  *(_QWORD *)(v40 + 40) = *(_QWORD *)(v40 + 112);
                  *(_QWORD *)(v40 + 48) = *(_QWORD *)(v40 + 120);
                  *(_DWORD *)(v40 + 56) = *(_DWORD *)(v40 + 128);
                  --v38;
                }
                while ( v38 );
                v28 = v84;
                LODWORD(v32) = *(_DWORD *)(v25 + 40);
                v23 = *(_QWORD *)(v25 + 32);
              }
              v41 = (unsigned int)(v32 - 1);
              *(_DWORD *)(v25 + 40) = v41;
              v42 = v23 + 72 * v41;
              v43 = *(_QWORD *)(v42 + 8);
              if ( v43 != v42 + 24 )
              {
                _libc_free(v43);
                v23 = *(_QWORD *)(v25 + 32);
              }
              if ( v85 != (__int64 *)(v23 + 72LL * *(unsigned int *)(v25 + 40)) )
              {
                v44 = 0x8E38E38E38E38E39LL * (((__int64)v85 - v23) >> 3);
                if ( *(_DWORD *)(v25 + 16) )
                {
                  v23 = *(_QWORD *)(v25 + 8);
                  v24 = v23 + 16LL * *(unsigned int *)(v25 + 24);
                  if ( v23 != v24 )
                  {
                    while ( 1 )
                    {
                      v77 = v23;
                      if ( *(_QWORD *)v23 != -4096 && *(_QWORD *)v23 != -8192 )
                        break;
                      v23 += 16;
                      if ( v24 == v23 )
                        goto LABEL_38;
                    }
                    if ( v23 != v24 )
                    {
                      do
                      {
                        v78 = *(unsigned int *)(v77 + 8);
                        v23 = v78;
                        if ( v44 < v78 )
                        {
                          v23 = (unsigned int)(v78 - 1);
                          *(_DWORD *)(v77 + 8) = v23;
                        }
                        v77 += 16;
                        if ( v77 == v24 )
                          break;
                        while ( 1 )
                        {
                          v23 = *(_QWORD *)v77;
                          if ( *(_QWORD *)v77 != -4096 && v23 != -8192 )
                            break;
                          v77 += 16;
                          if ( v24 == v77 )
                            goto LABEL_38;
                        }
                      }
                      while ( v24 != v77 );
                    }
                  }
                }
              }
            }
          }
        }
        else
        {
          v24 = 1;
          while ( v30 != -4096 )
          {
            a6 = (unsigned int)(v24 + 1);
            v23 = (unsigned int)a5 & ((_DWORD)v24 + (_DWORD)v23);
            v24 = v27 + 16LL * (unsigned int)v23;
            v30 = *(_QWORD *)v24;
            if ( v28 == *(_QWORD *)v24 )
              goto LABEL_26;
            v24 = (unsigned int)a6;
          }
        }
      }
LABEL_38:
      if ( v89 != a2 )
      {
        v86 = v25;
        v45 = a2;
        do
        {
          v46 = *(_DWORD *)(v45 + 24);
          v47 = *(_QWORD *)(v45 + 8);
          v93 = v28;
          if ( v46 )
          {
            v48 = v46 - 1;
            a5 = v48 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
            v49 = (__int64 *)(v47 + 8 * a5);
            v50 = *v49;
            if ( v28 == *v49 )
            {
LABEL_42:
              *v49 = -8192;
              v51 = *(unsigned int *)(v45 + 40);
              --*(_DWORD *)(v45 + 16);
              v52 = *(_QWORD **)(v45 + 32);
              ++*(_DWORD *)(v45 + 20);
              v53 = (__int64)&v52[v51];
              v54 = sub_28FEBC0(v52, v53, &v93);
              a6 = (__int64)(v54 + 1);
              if ( v54 + 1 != (_QWORD *)v53 )
              {
                memmove(v54, v54 + 1, v53 - a6);
                v55 = *(_DWORD *)(v45 + 40);
              }
              a5 = (unsigned int)(v55 - 1);
              *(_DWORD *)(v45 + 40) = a5;
            }
            else
            {
              v23 = 1;
              while ( v50 != -4096 )
              {
                v24 = (unsigned int)(v23 + 1);
                a5 = v48 & (unsigned int)(v23 + a5);
                v49 = (__int64 *)(v47 + 8LL * (unsigned int)a5);
                v50 = *v49;
                if ( v28 == *v49 )
                  goto LABEL_42;
                v23 = (unsigned int)v24;
              }
            }
          }
          v45 += 112;
        }
        while ( v45 != v89 );
        v25 = v86;
      }
      ++v91;
    }
    while ( v82 != v91 );
    v26 = v95;
  }
  if ( v26 )
  {
    v56 = *(_QWORD *)(v25 + 32);
    for ( i = v56 + 72LL * *(unsigned int *)(v25 + 40); i != v56; v56 += 72 )
    {
      if ( *(_DWORD *)(v56 + 16) > 1u )
      {
        *(_DWORD *)(v56 + 16) = 0;
        sub_28FF860(v56 + 8, *(unsigned __int8 **)v56, v23, v24, a5, a6);
      }
    }
  }
  if ( v94 != (__int64 *)v96 )
    _libc_free((unsigned __int64)v94);
}
