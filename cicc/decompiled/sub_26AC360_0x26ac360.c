// Function: sub_26AC360
// Address: 0x26ac360
//
void __fastcall sub_26AC360(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *v2; // r12
  _QWORD *v3; // rax
  __int64 v4; // rbx
  char v5; // dh
  __int64 *v6; // rsi
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // r14
  _QWORD *v11; // r12
  __int64 v12; // rbx
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r14
  _QWORD *v16; // r13
  __int64 v17; // rdx
  _QWORD *v18; // rdi
  __int64 v19; // r13
  char v20; // dh
  __int64 v21; // rcx
  char v22; // al
  int v23; // eax
  int v24; // eax
  unsigned int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rcx
  _QWORD *v31; // rbx
  __int64 v32; // r12
  __int64 *v33; // r14
  __int64 v34; // rax
  __int64 v35; // r15
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rsi
  int v39; // eax
  int v40; // eax
  unsigned int v41; // edx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rax
  unsigned int v46; // r9d
  __int64 v47; // rdx
  _QWORD *v48; // rax
  _QWORD *v49; // rdx
  __int64 v50; // rsi
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rax
  _QWORD **v54; // r13
  _QWORD **v55; // r14
  _QWORD *v56; // rdi
  __int64 v57; // rax
  __int64 v58; // r15
  __int64 v59; // r13
  unsigned __int64 v60; // rdi
  int v61; // eax
  __int64 v62; // rdi
  __int64 v63; // [rsp-E8h] [rbp-E8h]
  _QWORD *v64; // [rsp-D0h] [rbp-D0h]
  __int64 v65; // [rsp-C8h] [rbp-C8h]
  unsigned int v66; // [rsp-BCh] [rbp-BCh]
  _QWORD *v68; // [rsp-A8h] [rbp-A8h]
  __int64 *v69; // [rsp-A0h] [rbp-A0h]
  const char *v70; // [rsp-98h] [rbp-98h] BYREF
  __int16 v71; // [rsp-78h] [rbp-78h]
  _WORD *v72; // [rsp-68h] [rbp-68h] BYREF
  __int64 v73; // [rsp-60h] [rbp-60h]
  _WORD v74[44]; // [rsp-58h] [rbp-58h] BYREF

  v1 = *(_QWORD *)(a1 + 120);
  if ( v1 )
  {
    v2 = *(_QWORD **)(v1 + 48);
    v3 = (_QWORD *)v2[7];
    v68 = v2 + 6;
    if ( v2 + 6 != v3 )
    {
      if ( !v3 )
        BUG();
      if ( *((_BYTE *)v3 - 24) == 84 )
      {
        v66 = *(_DWORD *)(v1 + 72) + 1;
        if ( v66 < (*((_DWORD *)v3 - 5) & 0x7FFFFFFu) )
        {
          v4 = a1;
          v74[8] = 257;
          v6 = (__int64 *)sub_AA4FF0((__int64)v2);
          v7 = 0;
          if ( v6 )
            v7 = v5;
          v8 = 1;
          BYTE1(v8) = v7;
          *(_QWORD *)(*(_QWORD *)(a1 + 120) + 48LL) = sub_AA8550(v2, v6, v8, (__int64)&v72, 0);
          v9 = *(_QWORD *)(a1 + 120);
          v10 = (_QWORD *)v2[7];
          v65 = *(_QWORD *)(*(_QWORD *)(v9 + 48) + 56LL);
          v72 = v74;
          v73 = 0x400000000LL;
          if ( v10 == v68 )
          {
            v58 = *(_QWORD *)(v9 + 64);
            v59 = v58 + 8LL * *(unsigned int *)(v9 + 72);
            if ( v59 == v58 )
              return;
          }
          else
          {
            v64 = v2;
            v11 = v10;
            v12 = 1;
            do
            {
              if ( !v11 )
                BUG();
              if ( *((_BYTE *)v11 - 24) != 84 )
                break;
              v71 = 257;
              v13 = *(v11 - 2);
              v14 = sub_BD2DA0(80);
              v15 = v14;
              if ( v14 )
              {
                v16 = (_QWORD *)v14;
                sub_B44260(v14, v13, 55, 0x8000000u, 0, 0);
                *(_DWORD *)(v15 + 72) = v66;
                sub_BD6B50((unsigned __int8 *)v15, &v70);
                sub_BD2A10(v15, *(_DWORD *)(v15 + 72), 1);
              }
              else
              {
                v16 = 0;
              }
              v17 = v12;
              v18 = v16;
              LOBYTE(v12) = 1;
              v19 = (__int64)(v11 - 3);
              sub_B44220(v18, v65, v17);
              sub_BD84D0((__int64)(v11 - 3), v15);
              v21 = sub_AA4FF0(*(_QWORD *)(*(_QWORD *)(a1 + 120) + 48LL));
              v22 = 0;
              v65 = v21;
              if ( v21 )
                v22 = v20;
              BYTE1(v12) = v22;
              v23 = *(_DWORD *)(v15 + 4) & 0x7FFFFFF;
              if ( v23 == *(_DWORD *)(v15 + 72) )
              {
                sub_B48D90(v15);
                v23 = *(_DWORD *)(v15 + 4) & 0x7FFFFFF;
              }
              v24 = (v23 + 1) & 0x7FFFFFF;
              v25 = v24 | *(_DWORD *)(v15 + 4) & 0xF8000000;
              v26 = *(_QWORD *)(v15 - 8) + 32LL * (unsigned int)(v24 - 1);
              *(_DWORD *)(v15 + 4) = v25;
              if ( *(_QWORD *)v26 )
              {
                v27 = *(_QWORD *)(v26 + 8);
                **(_QWORD **)(v26 + 16) = v27;
                if ( v27 )
                  *(_QWORD *)(v27 + 16) = *(_QWORD *)(v26 + 16);
              }
              *(_QWORD *)v26 = v19;
              v28 = *(v11 - 1);
              *(_QWORD *)(v26 + 8) = v28;
              if ( v28 )
                *(_QWORD *)(v28 + 16) = v26 + 8;
              *(_QWORD *)(v26 + 16) = v11 - 1;
              *(v11 - 1) = v26;
              *(_QWORD *)(*(_QWORD *)(v15 - 8)
                        + 32LL * *(unsigned int *)(v15 + 72)
                        + 8LL * ((*(_DWORD *)(v15 + 4) & 0x7FFFFFFu) - 1)) = v64;
              v29 = *(_QWORD *)(a1 + 120);
              v30 = *(__int64 **)(v29 + 64);
              v69 = &v30[*(unsigned int *)(v29 + 72)];
              if ( v69 != v30 )
              {
                v63 = v12;
                v31 = v11;
                v32 = v15;
                v33 = *(__int64 **)(v29 + 64);
                do
                {
                  v34 = 0x1FFFFFFFE0LL;
                  v35 = *v33;
                  v36 = *(v31 - 4);
                  if ( (*((_DWORD *)v31 - 5) & 0x7FFFFFF) != 0 )
                  {
                    v37 = 0;
                    do
                    {
                      if ( v35 == *(_QWORD *)(v36 + 32LL * *((unsigned int *)v31 + 12) + 8 * v37) )
                      {
                        v34 = 32 * v37;
                        goto LABEL_31;
                      }
                      ++v37;
                    }
                    while ( (*((_DWORD *)v31 - 5) & 0x7FFFFFF) != (_DWORD)v37 );
                    v34 = 0x1FFFFFFFE0LL;
                  }
LABEL_31:
                  v38 = *(_QWORD *)(v36 + v34);
                  v39 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
                  if ( v39 == *(_DWORD *)(v32 + 72) )
                  {
                    sub_B48D90(v32);
                    v39 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
                  }
                  v40 = (v39 + 1) & 0x7FFFFFF;
                  v41 = v40 | *(_DWORD *)(v32 + 4) & 0xF8000000;
                  v42 = *(_QWORD *)(v32 - 8) + 32LL * (unsigned int)(v40 - 1);
                  *(_DWORD *)(v32 + 4) = v41;
                  if ( *(_QWORD *)v42 )
                  {
                    v43 = *(_QWORD *)(v42 + 8);
                    **(_QWORD **)(v42 + 16) = v43;
                    if ( v43 )
                      *(_QWORD *)(v43 + 16) = *(_QWORD *)(v42 + 16);
                  }
                  *(_QWORD *)v42 = v38;
                  if ( v38 )
                  {
                    v44 = *(_QWORD *)(v38 + 16);
                    *(_QWORD *)(v42 + 8) = v44;
                    if ( v44 )
                      *(_QWORD *)(v44 + 16) = v42 + 8;
                    *(_QWORD *)(v42 + 16) = v38 + 16;
                    *(_QWORD *)(v38 + 16) = v42;
                  }
                  *(_QWORD *)(*(_QWORD *)(v32 - 8)
                            + 32LL * *(unsigned int *)(v32 + 72)
                            + 8LL * ((*(_DWORD *)(v32 + 4) & 0x7FFFFFFu) - 1)) = v35;
                  if ( (*((_DWORD *)v31 - 5) & 0x7FFFFFF) != 0 )
                  {
                    v45 = 0;
                    while ( 1 )
                    {
                      v46 = v45;
                      if ( v35 == *(_QWORD *)(*(v31 - 4) + 32LL * *((unsigned int *)v31 + 12) + 8 * v45) )
                        break;
                      if ( (*((_DWORD *)v31 - 5) & 0x7FFFFFF) == (_DWORD)++v45 )
                        goto LABEL_67;
                    }
                  }
                  else
                  {
LABEL_67:
                    v46 = -1;
                  }
                  ++v33;
                  sub_B48BF0(v19, v46, 1);
                }
                while ( v69 != v33 );
                v11 = v31;
                v12 = v63;
              }
              v47 = 4LL * (*((_DWORD *)v11 - 5) & 0x7FFFFFF);
              if ( (*((_BYTE *)v11 - 17) & 0x40) != 0 )
              {
                v48 = (_QWORD *)*(v11 - 4);
                v49 = &v48[v47];
              }
              else
              {
                v48 = (_QWORD *)(v19 - v47 * 8);
                v49 = (_QWORD *)v19;
              }
              if ( v48 != v49 )
              {
                while ( 1 )
                {
                  v48 += 4;
                  if ( v48 == v49 )
                    break;
                  if ( *v48 != *(v48 - 4) )
                    goto LABEL_56;
                }
              }
              v50 = *(_QWORD *)*(v11 - 4);
              if ( v50 )
              {
                sub_BD84D0(v19, v50);
                v53 = (unsigned int)v73;
                if ( (unsigned __int64)(unsigned int)v73 + 1 > HIDWORD(v73) )
                {
                  sub_C8D5F0((__int64)&v72, v74, (unsigned int)v73 + 1LL, 8u, v51, v52);
                  v53 = (unsigned int)v73;
                }
                *(_QWORD *)&v72[4 * v53] = v19;
                LODWORD(v73) = v73 + 1;
              }
LABEL_56:
              v11 = (_QWORD *)v11[1];
            }
            while ( v68 != v11 );
            v54 = (_QWORD **)v72;
            v2 = v64;
            v4 = a1;
            v55 = (_QWORD **)&v72[4 * (unsigned int)v73];
            if ( v55 != (_QWORD **)v72 )
            {
              do
              {
                v56 = *v54++;
                sub_B43D60(v56);
              }
              while ( v55 != v54 );
            }
            v57 = *(_QWORD *)(a1 + 120);
            v58 = *(_QWORD *)(v57 + 64);
            v59 = v58 + 8LL * *(unsigned int *)(v57 + 72);
            if ( v59 == v58 )
              goto LABEL_65;
          }
          do
          {
            v60 = *(_QWORD *)(*(_QWORD *)v58 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v60 == *(_QWORD *)v58 + 48LL )
            {
              v62 = 0;
            }
            else
            {
              if ( !v60 )
                BUG();
              v61 = *(unsigned __int8 *)(v60 - 24);
              v62 = v60 - 24;
              if ( (unsigned int)(v61 - 30) >= 0xB )
                v62 = 0;
            }
            v58 += 8;
            sub_BD2ED0(v62, (__int64)v2, *(_QWORD *)(*(_QWORD *)(v4 + 120) + 48LL));
          }
          while ( v59 != v58 );
LABEL_65:
          if ( v72 != v74 )
            _libc_free((unsigned __int64)v72);
        }
      }
    }
  }
}
