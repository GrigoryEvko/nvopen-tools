// Function: sub_1DB2CC0
// Address: 0x1db2cc0
//
void __fastcall sub_1DB2CC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i *a6)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 (*v8)(void); // rax
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned __int64 *v11; // rdx
  __int64 v12; // r12
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rbx
  unsigned int v16; // r13d
  unsigned int v17; // esi
  char *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rax
  unsigned __int64 v22; // r14
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rcx
  signed __int64 v30; // r12
  __int64 v31; // rdi
  int *v32; // rcx
  int v33; // eax
  int v34; // ecx
  unsigned int v35; // ebx
  _QWORD *v36; // rax
  _QWORD *v37; // rdx
  _QWORD *v38; // r11
  __int64 v39; // r13
  __int64 v40; // r11
  __int64 v41; // rbx
  __int64 v42; // r14
  __int64 v43; // r13
  __int64 v44; // r12
  unsigned __int64 v45; // rdi
  __int64 v46; // rdi
  __int64 v47; // rax
  int v48; // edx
  _QWORD *v49; // rdx
  _QWORD *v50; // r9
  unsigned int v51; // esi
  __int64 v52; // rdi
  __int64 v53; // rcx
  __int64 v54; // rbx
  __int64 v55; // rsi
  __int64 v56; // [rsp+18h] [rbp-188h]
  __int64 v58; // [rsp+30h] [rbp-170h]
  __int64 v59; // [rsp+48h] [rbp-158h]
  __int64 v60; // [rsp+50h] [rbp-150h]
  unsigned __int64 v61; // [rsp+58h] [rbp-148h]
  __int64 v62; // [rsp+60h] [rbp-140h]
  unsigned int v63; // [rsp+60h] [rbp-140h]
  __int64 v64; // [rsp+68h] [rbp-138h]
  __int64 v65; // [rsp+70h] [rbp-130h]
  __int64 v66; // [rsp+78h] [rbp-128h]
  int v67; // [rsp+80h] [rbp-120h]
  char v68; // [rsp+84h] [rbp-11Ch]
  __int64 v69; // [rsp+88h] [rbp-118h]
  unsigned int v70; // [rsp+88h] [rbp-118h]
  __int64 v71; // [rsp+88h] [rbp-118h]
  _QWORD v72[2]; // [rsp+90h] [rbp-110h] BYREF
  int v73; // [rsp+A0h] [rbp-100h]
  __int64 v74; // [rsp+B0h] [rbp-F0h]
  _BYTE *v75; // [rsp+B8h] [rbp-E8h] BYREF
  __int64 v76; // [rsp+C0h] [rbp-E0h]
  _BYTE v77[72]; // [rsp+C8h] [rbp-D8h] BYREF
  __int64 v78; // [rsp+110h] [rbp-90h]
  char *v79; // [rsp+118h] [rbp-88h] BYREF
  __int64 v80; // [rsp+120h] [rbp-80h]
  __int64 v81; // [rsp+128h] [rbp-78h] BYREF
  __int64 v82; // [rsp+130h] [rbp-70h]

  v6 = *(_QWORD *)(a1 + 232);
  v58 = v6;
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 120);
    if ( v7 )
    {
      v64 = 0;
      v8 = *(__int64 (**)(void))(**(_QWORD **)(v7 + 16) + 40LL);
      if ( v8 != sub_1D00B00 )
        v64 = v8();
      v72[0] = 0;
      v72[1] = 0;
      v73 = 0;
      v9 = *(unsigned int *)(v58 + 160);
      if ( (_DWORD)v9 )
      {
        v59 = 0;
        v56 = 8 * v9;
        while ( 1 )
        {
          sub_1DB1BC0(*(_QWORD *)(*(_QWORD *)(v58 + 152) + v59), a2, *(_QWORD *)(v58 + 136), (__int64)v72, a5, a6);
          v11 = (unsigned __int64 *)&v81;
          v12 = *(_QWORD *)(v58 + 128);
          v13 = *(_QWORD *)(*(_QWORD *)(v58 + 152) + v59);
          v14 = *(_QWORD *)(v58 + 136);
          v79 = (char *)&v81;
          v80 = 0x400000000LL;
          v15 = *(_QWORD *)(a2 + 256);
          v16 = *(_DWORD *)(v13 + 296);
          v66 = v14;
          v78 = v13 + 216;
          if ( v16 )
          {
            LODWORD(v80) = 1;
            v17 = 4;
            v18 = (char *)&v81;
            v81 = v13 + 224;
            LODWORD(v10) = 1;
            v82 = *(unsigned int *)(v13 + 300);
            v19 = 0;
            while ( 1 )
            {
              v20 = *(_QWORD *)(*(_QWORD *)&v18[16 * v19] + 8LL * *(unsigned int *)&v18[16 * v19 + 12]);
              v21 = v20 & 0x3F;
              v22 = v20 & 0xFFFFFFFFFFFFFFC0LL;
              a5 = v21 + 1;
              if ( (unsigned int)v10 >= v17 )
              {
                v71 = v21 + 1;
                sub_16CD150((__int64)&v79, &v81, 0, 16, a5, (int)a6);
                v18 = v79;
                a5 = v71;
              }
              v11 = (unsigned __int64 *)&v18[16 * (unsigned int)v80];
              *v11 = v22;
              v11[1] = a5;
              v19 = (unsigned int)v80;
              v10 = (unsigned int)(v80 + 1);
              LODWORD(v80) = v80 + 1;
              if ( v16 <= (unsigned int)v19 )
                break;
              v18 = v79;
              v17 = HIDWORD(v80);
            }
            v74 = v78;
            v75 = v77;
            v76 = 0x400000000LL;
            if ( !(_DWORD)v10 )
              goto LABEL_14;
          }
          else
          {
            v81 = v13 + 216;
            v46 = *(unsigned int *)(v13 + 300);
            v74 = v13 + 216;
            v82 = v46;
            v75 = v77;
            LODWORD(v80) = 1;
            v76 = 0x400000000LL;
          }
          sub_1DA8090((__int64)&v75, &v79, (__int64)v11, v10, a5, (int)a6);
LABEL_14:
          if ( v79 != (char *)&v81 )
            _libc_free((unsigned __int64)v79);
          v23 = (unsigned int)v76;
          v24 = (unsigned __int64)v75;
          if ( (_DWORD)v76 )
          {
            v25 = v12;
            v65 = v15 + 320;
            do
            {
              if ( *(_DWORD *)(v24 + 12) >= *(_DWORD *)(v24 + 8) )
                break;
              v26 = v24 + 16 * v23 - 16;
              v27 = *(_QWORD *)v26;
              v28 = *(unsigned int *)(v26 + 12);
              v29 = 16 * v28;
              v30 = *(_QWORD *)(v27 + 16 * v28);
              if ( *(_DWORD *)(v74 + 80) )
              {
                v31 = *(_QWORD *)(v27 + v29 + 8);
                v32 = (int *)(v27 + 4 * (v28 + 36));
                v69 = v31;
              }
              else
              {
                v54 = *(_QWORD *)(v27 + v29 + 8);
                v32 = (int *)(v27 + 4 * (v28 + 16));
                v69 = v54;
              }
              v33 = *v32 & 0x7FFFFFFF;
              v34 = *v32;
              v68 = 0;
              v35 = v33 | v34 & 0x80000000;
              v67 = v34 & 0x7FFFFFFF;
              if ( (v33 & 0x7FFFFFFF) != 0x7FFFFFFF )
                v68 = (*(_QWORD *)(v72[0] + 8LL * ((v34 & 0x7FFFFFFFu) >> 6)) & (1LL << v34)) != 0;
              if ( *(_QWORD *)(v13 + 384) )
              {
                v49 = *(_QWORD **)(v13 + 360);
                if ( v49 )
                {
                  v50 = (_QWORD *)(v13 + 352);
                  v38 = (_QWORD *)(v30 & 0xFFFFFFFFFFFFFFF8LL);
                  v51 = *(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v30 >> 1) & 3;
                  do
                  {
                    while ( 1 )
                    {
                      v52 = v49[2];
                      v53 = v49[3];
                      if ( (*(_DWORD *)((v49[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v49[4] >> 1) & 3) >= v51 )
                        break;
                      v49 = (_QWORD *)v49[3];
                      if ( !v53 )
                        goto LABEL_54;
                    }
                    v50 = v49;
                    v49 = (_QWORD *)v49[2];
                  }
                  while ( v52 );
LABEL_54:
                  if ( (_QWORD *)(v13 + 352) != v50
                    && v51 >= (*(_DWORD *)((v50[4] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                             | (unsigned int)((__int64)v50[4] >> 1) & 3) )
                  {
                    goto LABEL_30;
                  }
                }
              }
              else
              {
                v36 = *(_QWORD **)(v13 + 312);
                v37 = &v36[*(unsigned int *)(v13 + 320)];
                if ( v36 != v37 )
                {
                  while ( *v36 != v30 )
                  {
                    if ( v37 == ++v36 )
                      goto LABEL_31;
                  }
                  if ( v36 != v37 )
                  {
                    v38 = (_QWORD *)(v30 & 0xFFFFFFFFFFFFFFF8LL);
LABEL_30:
                    v30 = *v38 & 0xFFFFFFFFFFFFFFF8LL | v30 & 6;
                  }
                }
              }
LABEL_31:
              v62 = *(_QWORD *)(v25 + 272);
              v39 = sub_1DA9310(v62, v30);
              v60 = *(_QWORD *)(*(_QWORD *)(v62 + 392) + 16LL * *(unsigned int *)(v39 + 48) + 8);
              sub_1DA8B50((__int64 *)v13, v39, v30, v69, v35, v68, v25, v64, v66);
              v40 = v60;
              v61 = v69 & 0xFFFFFFFFFFFFFFF8LL;
              v63 = (v69 >> 1) & 3;
              if ( (*(_DWORD *)((v60 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v60 >> 1) & 3) < (*(_DWORD *)((v69 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v63) )
              {
                a5 = v35;
                v41 = v25;
                v42 = v39;
                v43 = v69;
                while ( 1 )
                {
                  v42 = *(_QWORD *)(v42 + 8);
                  if ( v65 == v42 )
                    break;
                  v70 = v67 & 0x7FFFFFFF | a5 & 0x80000000;
                  v44 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v41 + 272) + 392LL) + 16LL * *(unsigned int *)(v42 + 48) + 8);
                  sub_1DA8B50((__int64 *)v13, v42, v40, v43, v70, v68, v41, v64, v66);
                  a5 = v70;
                  if ( (*(_DWORD *)(v61 + 24) | v63) <= (*(_DWORD *)((v44 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                       | (unsigned int)(v44 >> 1) & 3) )
                  {
                    v25 = v41;
                    goto LABEL_46;
                  }
                  v40 = v44;
                }
LABEL_36:
                v24 = (unsigned __int64)v75;
                break;
              }
              if ( v65 == v39 )
                goto LABEL_36;
LABEL_46:
              v47 = (__int64)&v75[16 * (unsigned int)v76 - 16];
              v48 = *(_DWORD *)(v47 + 12) + 1;
              *(_DWORD *)(v47 + 12) = v48;
              v23 = (unsigned int)v76;
              v24 = (unsigned __int64)v75;
              if ( v48 == *(_DWORD *)&v75[16 * (unsigned int)v76 - 8] )
              {
                v55 = *(unsigned int *)(v74 + 80);
                if ( (_DWORD)v55 )
                {
                  sub_39460A0(&v75, v55);
                  v23 = (unsigned int)v76;
                  v24 = (unsigned __int64)v75;
                }
              }
            }
            while ( (_DWORD)v23 );
          }
          if ( (_BYTE *)v24 != v77 )
            _libc_free(v24);
          v59 += 8;
          if ( v56 == v59 )
          {
            v45 = v72[0];
            goto LABEL_41;
          }
        }
      }
      v45 = 0;
LABEL_41:
      *(_BYTE *)(v58 + 144) = 1;
      _libc_free(v45);
    }
  }
}
