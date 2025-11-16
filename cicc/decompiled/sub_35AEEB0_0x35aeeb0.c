// Function: sub_35AEEB0
// Address: 0x35aeeb0
//
void __fastcall sub_35AEEB0(__int64 a1, __int64 *a2, unsigned int *a3, unsigned int *a4)
{
  __int64 v5; // rbx
  unsigned __int16 *v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  int v9; // r15d
  unsigned __int16 *v10; // r13
  unsigned int v11; // r12d
  unsigned __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 i; // rsi
  __int64 v15; // rcx
  int v16; // r12d
  unsigned int v17; // edi
  __int16 *v18; // rcx
  __int16 *v19; // rax
  int v20; // ecx
  unsigned __int16 j; // cx
  __int64 v22; // r9
  unsigned __int64 v23; // rcx
  int v24; // ecx
  __int64 v25; // rdi
  __int64 (*v26)(); // rax
  __int64 v27; // rdx
  _QWORD *v28; // r14
  __int64 (__fastcall *v29)(__int64); // r11
  __int64 (*v30)(); // rax
  char *v31; // r11
  __int64 v32; // rsi
  __int64 (__fastcall *v33)(__int64, unsigned int *); // rax
  __int64 v34; // r12
  char *v35; // r15
  __int64 v36; // r13
  unsigned int v37; // r14d
  __int64 *v38; // r9
  __int64 (*v39)(); // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  unsigned int v43; // eax
  char *v44; // rdi
  unsigned __int64 v45; // rbx
  __int64 v46; // rax
  unsigned __int64 v47; // rsi
  unsigned __int64 v48; // rbx
  unsigned __int64 v49; // rdx
  char *v50; // rax
  unsigned __int64 v51; // r8
  size_t v52; // rdx
  char v53; // al
  int v54; // eax
  unsigned __int8 v55; // dl
  __int64 v56; // rax
  char v57; // al
  __int64 v58; // rax
  __int64 *v59; // [rsp+0h] [rbp-F0h]
  _BYTE *v60; // [rsp+10h] [rbp-E0h]
  __int64 v61; // [rsp+18h] [rbp-D8h]
  unsigned int v65; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v66; // [rsp+44h] [rbp-ACh] BYREF
  int v67; // [rsp+4Ch] [rbp-A4h]
  char *v68; // [rsp+50h] [rbp-A0h] BYREF
  char *v69; // [rsp+58h] [rbp-98h]
  char *v70; // [rsp+60h] [rbp-90h]
  void *v71; // [rsp+70h] [rbp-80h] BYREF
  __int64 v72; // [rsp+78h] [rbp-78h]
  _BYTE v73[48]; // [rsp+80h] [rbp-70h] BYREF
  int v74; // [rsp+B0h] [rbp-40h]

  if ( *((_DWORD *)a2 + 16) )
  {
    v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 200LL))(*(_QWORD *)(a1 + 16));
    v6 = sub_2EBFBC0(*(_QWORD **)(a1 + 32));
    v9 = *((_DWORD *)a2 + 16);
    v10 = v6;
    v71 = v73;
    v11 = (unsigned int)(v9 + 63) >> 6;
    v72 = 0x600000000LL;
    if ( v11 > 6 )
    {
      sub_C8D5F0((__int64)&v71, v73, v11, 8u, v7, v8);
      memset(v71, 0, 8LL * v11);
      LODWORD(v72) = (unsigned int)(v9 + 63) >> 6;
    }
    else
    {
      if ( v11 )
      {
        v52 = 8LL * v11;
        if ( v52 )
          memset(v73, 0, v52);
      }
      LODWORD(v72) = (unsigned int)(v9 + 63) >> 6;
    }
    v74 = v9;
    v12 = *v10;
    v13 = 0;
    for ( i = 1; (_WORD)v12; v13 = (unsigned int)(v13 + 1) )
    {
      *(_QWORD *)((char *)v71 + ((v12 >> 3) & 0x1FF8)) |= 1LL << v12;
      v12 = v10[(unsigned int)(v13 + 1)];
    }
    v68 = 0;
    v69 = 0;
    v70 = 0;
    v15 = *v10;
    if ( (_WORD)v15 )
    {
      v16 = 0;
      do
      {
        v17 = (unsigned __int16)v15;
        v13 = *a2;
        i = (unsigned __int16)v15 >> 6;
        if ( (*(_QWORD *)(*a2 + 8 * i) & (1LL << v15)) != 0 )
        {
          v18 = (__int16 *)(*(_QWORD *)(v5 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v5 + 8) + 24 * v15 + 8));
          v19 = v18 + 1;
          v20 = *v18;
          i = v17 + v20;
          if ( (_WORD)v20 )
          {
            for ( j = v17 + v20; ; j = i )
            {
              v22 = 1LL << j;
              v23 = (unsigned __int64)j >> 6;
              if ( (*(_QWORD *)(v13 + 8 * v23) & v22) != 0 && (*((_QWORD *)v71 + v23) & v22) != 0 )
                break;
              v24 = *v19++;
              if ( !(_WORD)v24 )
                goto LABEL_16;
              i = (unsigned int)(v24 + i);
            }
          }
          else
          {
LABEL_16:
            v66 = v17;
            i = (__int64)v69;
            LOWORD(v67) = 1;
            if ( v69 == v70 )
            {
              sub_35AED10((unsigned __int64 *)&v68, v69, (__int64)&v66);
            }
            else
            {
              if ( v69 )
              {
                *(_QWORD *)v69 = v66;
                *(_DWORD *)(i + 8) = v67;
                i = (__int64)v69;
              }
              i += 12;
              v69 = (char *)i;
            }
          }
        }
        v15 = v10[++v16];
      }
      while ( (_WORD)v15 );
    }
    v25 = *(_QWORD *)(a1 + 16);
    v26 = *(__int64 (**)())(*(_QWORD *)v25 + 136LL);
    if ( v26 == sub_2DD19D0 )
      BUG();
    v60 = (_BYTE *)((__int64 (__fastcall *)(__int64, __int64, __int64))v26)(v25, i, v13);
    v28 = *(_QWORD **)(a1 + 48);
    v29 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v60 + 40LL);
    if ( v29 == sub_2FDBDE0 )
    {
      v30 = *(__int64 (**)())(*(_QWORD *)v60 + 48LL);
      if ( v30 == sub_2FDBB70 )
      {
LABEL_24:
        v31 = v68;
        v32 = (__int64)v69;
        if ( v69 == v68 )
        {
          v44 = v69;
          if ( !v69 )
          {
LABEL_53:
            if ( v71 != v73 )
              _libc_free((unsigned __int64)v71);
            return;
          }
LABEL_52:
          j_j___libc_free_0((unsigned __int64)v44);
          goto LABEL_53;
        }
        v33 = *(__int64 (__fastcall **)(__int64, unsigned int *))(*(_QWORD *)v60 + 56LL);
        if ( v33 == sub_2FDBB80 )
        {
          v65 = 0;
          v34 = 0;
        }
        else
        {
          v58 = v33((__int64)v60, &v65);
          v31 = v68;
          v34 = v58;
          v44 = v68;
          v32 = (__int64)v69;
          if ( v69 == v68 )
          {
LABEL_40:
            v45 = v31 - v44;
            if ( v31 == v44 )
            {
              v47 = 0;
            }
            else
            {
              if ( v45 > 0x7FFFFFFFFFFFFFF8LL )
                sub_4261EA(v44, v32, v27);
              v46 = sub_22077B0(v31 - v44);
              v31 = v69;
              v44 = v68;
              v47 = v46;
            }
            v48 = v47 + v45;
            v49 = v47;
            if ( v31 != v44 )
            {
              v50 = v44;
              do
              {
                if ( v49 )
                {
                  *(_QWORD *)v49 = *(_QWORD *)v50;
                  *(_DWORD *)(v49 + 8) = *((_DWORD *)v50 + 2);
                }
                v50 += 12;
                v49 += 12LL;
              }
              while ( v31 != v50 );
              v49 = v47 + 4 * ((unsigned __int64)(v31 - 12 - v44) >> 2) + 12;
            }
            v51 = v28[12];
            v28[12] = v47;
            v28[13] = v49;
            v28[14] = v48;
            if ( v51 )
            {
              j_j___libc_free_0(v51);
              v44 = v68;
            }
            if ( !v44 )
              goto LABEL_53;
            goto LABEL_52;
          }
        }
        v61 = (__int64)v28;
        v35 = v31;
        v36 = v32;
        do
        {
          if ( !v35[9] )
          {
            v37 = *(_DWORD *)v35;
            v38 = sub_2FF6500(v5, *(_DWORD *)v35, 1);
            v39 = *(__int64 (**)())(*(_QWORD *)v5 + 520LL);
            if ( v39 == sub_2FF52C0
              || (v59 = v38,
                  v32 = a1,
                  v53 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64 *))v39)(v5, a1, v37, &v66),
                  v38 = v59,
                  !v53) )
            {
              v40 = v34;
              v41 = v34 + 16LL * v65;
              if ( v34 != v41 )
              {
                do
                {
                  if ( *(_DWORD *)v40 == v37 )
                    break;
                  v40 += 16;
                }
                while ( v40 != v41 );
              }
              v42 = *(_QWORD *)(v5 + 312)
                  + 16LL
                  * (*(unsigned __int16 *)(*v38 + 24)
                   + *(_DWORD *)(v5 + 328)
                   * (unsigned int)((__int64)(*(_QWORD *)(v5 + 288) - *(_QWORD *)(v5 + 280)) >> 3));
              v32 = *(_DWORD *)(v42 + 4) >> 3;
              if ( v40 == v41 )
              {
                v55 = -1;
                LODWORD(v56) = *(_DWORD *)(v42 + 8) >> 3;
                if ( (_DWORD)v56 )
                {
                  _BitScanReverse64((unsigned __int64 *)&v56, (unsigned int)v56);
                  v55 = 63 - (v56 ^ 0x3F);
                }
                if ( v60[12] <= v55 )
                  v55 = v60[12];
                v43 = sub_2E77BD0(v61, v32, v55, 1u, 0, 0);
                if ( *a3 > v43 )
                  *a3 = v43;
                if ( v43 > *a4 )
                  *a4 = v43;
              }
              else
              {
                v43 = sub_2E77A40(v61, v32, *(_QWORD *)(v40 + 8), 0);
              }
              *((_DWORD *)v35 + 1) = v43;
              v35[9] = 0;
            }
            else
            {
              v54 = v66;
              v35[9] = 0;
              *((_DWORD *)v35 + 1) = v54;
            }
          }
          v35 += 12;
        }
        while ( (char *)v36 != v35 );
        v28 = (_QWORD *)v61;
LABEL_39:
        v31 = v69;
        v44 = v68;
        goto LABEL_40;
      }
      v32 = a1;
      v57 = ((__int64 (__fastcall *)(_BYTE *, __int64, __int64, char **))v30)(v60, a1, v5, &v68);
    }
    else
    {
      v32 = a1;
      v57 = ((__int64 (__fastcall *)(_BYTE *, __int64, __int64, char **, unsigned int *, unsigned int *))v29)(
              v60,
              a1,
              v5,
              &v68,
              a3,
              a4);
    }
    if ( v57 )
      goto LABEL_39;
    goto LABEL_24;
  }
}
