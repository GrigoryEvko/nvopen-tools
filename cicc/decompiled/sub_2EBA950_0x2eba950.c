// Function: sub_2EBA950
// Address: 0x2eba950
//
void __fastcall sub_2EBA950(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rsi
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned __int64 *v15; // r12
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // eax
  unsigned __int64 *v21; // r13
  __int64 v22; // rdx
  __int64 v23; // r15
  int v24; // r10d
  __int64 v25; // rax
  int v26; // r10d
  __int64 v27; // r14
  __int64 v28; // rax
  int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 *v33; // r14
  char *v34; // r10
  unsigned int v35; // eax
  __int64 v36; // rax
  unsigned __int64 v37; // r11
  _QWORD *v38; // rax
  __int64 v39; // r15
  unsigned __int64 *v40; // rax
  __int64 v41; // rbx
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  unsigned __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 *v50; // rdx
  __int64 *v51; // rbx
  __int64 v52; // rdx
  unsigned int v53; // eax
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // rdx
  unsigned int v57; // eax
  __int64 v58; // r14
  __int64 v59; // r12
  _QWORD *v60; // rdi
  __int64 v61; // rsi
  _QWORD *v62; // rax
  int v63; // r9d
  size_t v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rsi
  _BYTE *v68; // rbx
  unsigned __int64 v69; // r12
  unsigned __int64 v70; // rdi
  __int64 v71; // [rsp+8h] [rbp-1508h]
  char *v72; // [rsp+10h] [rbp-1500h]
  __int64 *v73; // [rsp+18h] [rbp-14F8h]
  unsigned int v74; // [rsp+38h] [rbp-14D8h]
  unsigned int v75; // [rsp+3Ch] [rbp-14D4h]
  unsigned int v76; // [rsp+40h] [rbp-14D0h]
  __int64 v77; // [rsp+40h] [rbp-14D0h]
  int v78; // [rsp+48h] [rbp-14C8h]
  __int64 *v79; // [rsp+48h] [rbp-14C8h]
  unsigned __int64 v80; // [rsp+48h] [rbp-14C8h]
  __int64 *v81; // [rsp+50h] [rbp-14C0h] BYREF
  int v82; // [rsp+58h] [rbp-14B8h]
  char v83; // [rsp+60h] [rbp-14B0h] BYREF
  _QWORD *v84; // [rsp+A0h] [rbp-1470h] BYREF
  __int64 v85; // [rsp+A8h] [rbp-1468h]
  _QWORD v86[128]; // [rsp+B0h] [rbp-1460h] BYREF
  unsigned __int64 v87[2]; // [rsp+4B0h] [rbp-1060h] BYREF
  _QWORD v88[64]; // [rsp+4C0h] [rbp-1050h] BYREF
  _BYTE *v89; // [rsp+6C0h] [rbp-E50h]
  __int64 v90; // [rsp+6C8h] [rbp-E48h]
  _BYTE v91[3584]; // [rsp+6D0h] [rbp-E40h] BYREF
  __int64 v92; // [rsp+14D0h] [rbp-40h]

  v6 = a1;
  v7 = sub_2EB3BB0(a1, a3, a4);
  v10 = (_QWORD *)v7;
  if ( v7 )
  {
    v11 = (unsigned int)(*(_DWORD *)(v7 + 24) + 1);
    v12 = *(_DWORD *)(v7 + 24) + 1;
  }
  else
  {
    v11 = 0;
    v12 = 0;
  }
  if ( v12 >= *(_DWORD *)(a1 + 56) )
    BUG();
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v11);
  v73 = *(__int64 **)(v13 + 8);
  if ( v73 )
  {
    v14 = *(_DWORD *)(v13 + 16);
    v92 = a2;
    v15 = v87;
    v75 = v14;
    v89 = v91;
    v87[0] = (unsigned __int64)v88;
    v90 = 0x4000000000LL;
    v87[1] = 0x4000000001LL;
    v88[0] = 0;
    v84 = v86;
    v86[0] = v10;
    v86[1] = 0;
    v85 = 0x4000000001LL;
    *(_DWORD *)(sub_2EB5B40((__int64)v87, (__int64)v10, v11, (__int64)v86, v8, v9) + 4) = 0;
    v20 = v85;
    if ( (_DWORD)v85 )
    {
      v76 = 0;
      v21 = v87;
      do
      {
        while ( 1 )
        {
          v22 = (__int64)&v84[2 * v20 - 2];
          v23 = *(_QWORD *)v22;
          v24 = *(_DWORD *)(v22 + 8);
          LODWORD(v85) = v20 - 1;
          v10 = (_QWORD *)v23;
          v78 = v24;
          v25 = sub_2EB5B40((__int64)v21, v23, v22, (__int64)v84, v18, v19);
          v26 = v78;
          v27 = v25;
          v28 = *(unsigned int *)(v25 + 32);
          v17 = *(unsigned int *)(v27 + 36);
          if ( v28 + 1 > v17 )
          {
            v10 = (_QWORD *)(v27 + 40);
            sub_C8D5F0(v27 + 24, (const void *)(v27 + 40), v28 + 1, 4u, v18, v19);
            v28 = *(unsigned int *)(v27 + 32);
            v26 = v78;
          }
          v16 = *(_QWORD *)(v27 + 24);
          *(_DWORD *)(v16 + 4 * v28) = v26;
          v29 = *(_DWORD *)v27;
          ++*(_DWORD *)(v27 + 32);
          if ( !v29 )
          {
            ++v76;
            *(_DWORD *)(v27 + 4) = v26;
            *(_DWORD *)(v27 + 12) = v76;
            *(_DWORD *)(v27 + 8) = v76;
            *(_DWORD *)v27 = v76;
            sub_2E6D5A0((__int64)v21, v23, v16, v17, v18, v19);
            v10 = (_QWORD *)v23;
            sub_2EB52F0(&v81, v23, v92, v30, v31, v32);
            v33 = v81;
            v34 = (char *)&v81[v82];
            if ( v81 != (__int64 *)v34 )
            {
              v19 = (__int64)v21;
              v18 = v76;
              do
              {
                v39 = *v33;
                if ( *v33 )
                {
                  v16 = (unsigned int)(*(_DWORD *)(v39 + 24) + 1);
                  v35 = *(_DWORD *)(v39 + 24) + 1;
                }
                else
                {
                  v35 = 0;
                  v16 = 0;
                }
                if ( v35 >= *(_DWORD *)(a1 + 56) )
                  BUG();
                if ( v75 < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v16) + 16LL) )
                {
                  v17 = HIDWORD(v85);
                  v36 = (unsigned int)v85;
                  v37 = v4 & 0xFFFFFFFF00000000LL | (unsigned int)v18;
                  v16 = (unsigned int)v85 + 1LL;
                  v4 = v37;
                  if ( v16 > HIDWORD(v85) )
                  {
                    v10 = v86;
                    v71 = v19;
                    v72 = v34;
                    v74 = v18;
                    v80 = v37;
                    sub_C8D5F0((__int64)&v84, v86, v16, 0x10u, v18, v19);
                    v36 = (unsigned int)v85;
                    v19 = v71;
                    v34 = v72;
                    v18 = v74;
                    v37 = v80;
                  }
                  v38 = &v84[2 * v36];
                  *v38 = v39;
                  v38[1] = v37;
                  LODWORD(v85) = v85 + 1;
                }
                ++v33;
              }
              while ( v34 != (char *)v33 );
              v34 = (char *)v81;
              v21 = (unsigned __int64 *)v19;
            }
            if ( v34 != &v83 )
              break;
          }
          v20 = v85;
          if ( !(_DWORD)v85 )
            goto LABEL_25;
        }
        _libc_free((unsigned __int64)v34);
        v20 = v85;
      }
      while ( (_DWORD)v85 );
LABEL_25:
      v40 = v21;
      v6 = a1;
      v15 = v40;
    }
    if ( v84 != v86 )
      _libc_free((unsigned __int64)v84);
    sub_2EB5CF0((__int64)v15, (__int64)v10, v16, v17, v18, v19);
    v41 = *v73;
    *(_QWORD *)(sub_2EB5B40((__int64)v15, *(_QWORD *)(v87[0] + 8), v42, v43, v44, v45) + 16) = v41;
    v46 = sub_2E6E010(v15, 1);
    v79 = v50;
    v51 = (__int64 *)v46;
    if ( (__int64 *)v46 != v50 )
    {
      v77 = (__int64)v15;
      do
      {
        v67 = *v51;
        if ( *v51 )
        {
          v52 = (unsigned int)(*(_DWORD *)(v67 + 24) + 1);
          v53 = *(_DWORD *)(v67 + 24) + 1;
        }
        else
        {
          v52 = 0;
          v53 = 0;
        }
        v54 = 0;
        if ( v53 < *(_DWORD *)(v6 + 56) )
          v54 = *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8 * v52);
        v55 = *(_QWORD *)(sub_2EB5B40(v77, v67, v52, v47, v48, v49) + 16);
        if ( v55 )
        {
          v56 = (unsigned int)(*(_DWORD *)(v55 + 24) + 1);
          v57 = *(_DWORD *)(v55 + 24) + 1;
        }
        else
        {
          v56 = 0;
          v57 = 0;
        }
        v58 = 0;
        if ( v57 < *(_DWORD *)(v6 + 56) )
          v58 = *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8 * v56);
        v59 = *(_QWORD *)(v54 + 8);
        if ( v58 != v59 )
        {
          v84 = (_QWORD *)v54;
          v60 = *(_QWORD **)(v59 + 24);
          v61 = (__int64)&v60[*(unsigned int *)(v59 + 32)];
          v62 = sub_2EB30D0(v60, v61, (__int64 *)&v84);
          if ( v62 + 1 != (_QWORD *)v61 )
          {
            v64 = v61 - (_QWORD)(v62 + 1);
            v61 = (__int64)(v62 + 1);
            memmove(v62, v62 + 1, v64);
            v63 = *(_DWORD *)(v59 + 32);
          }
          v49 = (unsigned int)(v63 - 1);
          *(_DWORD *)(v59 + 32) = v49;
          *(_QWORD *)(v54 + 8) = v58;
          v65 = *(unsigned int *)(v58 + 32);
          v47 = *(unsigned int *)(v58 + 36);
          if ( v65 + 1 > v47 )
          {
            v61 = v58 + 40;
            sub_C8D5F0(v58 + 24, (const void *)(v58 + 40), v65 + 1, 8u, v48, v49);
            v65 = *(unsigned int *)(v58 + 32);
          }
          v66 = *(_QWORD *)(v58 + 24);
          *(_QWORD *)(v66 + 8 * v65) = v54;
          ++*(_DWORD *)(v58 + 32);
          if ( *(_DWORD *)(v54 + 16) != *(_DWORD *)(*(_QWORD *)(v54 + 8) + 16LL) + 1 )
            sub_2EB3540(v54, v61, v66, v47, v48, v49);
        }
        ++v51;
      }
      while ( v79 != v51 );
    }
    v68 = v89;
    v69 = (unsigned __int64)&v89[56 * (unsigned int)v90];
    if ( v89 != (_BYTE *)v69 )
    {
      do
      {
        v69 -= 56LL;
        v70 = *(_QWORD *)(v69 + 24);
        if ( v70 != v69 + 40 )
          _libc_free(v70);
      }
      while ( v68 != (_BYTE *)v69 );
      v69 = (unsigned __int64)v89;
    }
    if ( (_BYTE *)v69 != v91 )
      _libc_free(v69);
    if ( (_QWORD *)v87[0] != v88 )
      _libc_free(v87[0]);
  }
  else
  {
    sub_2EBA1B0(a1, a2);
  }
}
