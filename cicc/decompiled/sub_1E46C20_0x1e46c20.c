// Function: sub_1E46C20
// Address: 0x1e46c20
//
void __fastcall sub_1E46C20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 (*v5)(void); // rax
  _QWORD *v6; // rax
  __int64 v7; // r15
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r12
  unsigned int v11; // r15d
  __int64 (*v12)(); // rax
  unsigned int v13; // ecx
  __int16 v14; // ax
  __int64 v15; // rsi
  int v16; // r15d
  __int64 v17; // rbx
  int v18; // eax
  __int64 v19; // r9
  __int64 v20; // r15
  _QWORD *v21; // r14
  int *v22; // r8
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r12
  unsigned __int8 v26; // al
  __int64 v27; // r13
  _DWORD *v28; // rax
  _DWORD *v29; // rdx
  _BYTE *v30; // rsi
  __int64 v31; // rdx
  __int64 (*v32)(); // rax
  __int64 v33; // rax
  __int64 v34; // r9
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int16 v37; // r10
  _WORD *v38; // rax
  __int16 *v39; // rdx
  __int64 v40; // r8
  unsigned __int16 v41; // bx
  int *v42; // r12
  __int64 v43; // r10
  __int16 *v44; // r13
  _DWORD *v45; // rax
  _DWORD *v46; // rcx
  __int16 v47; // ax
  __int64 v48; // rdi
  int *v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rsi
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // r11
  __int64 v56; // rax
  int *v57; // rsi
  __int64 v58; // rcx
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v63; // [rsp+38h] [rbp-108h]
  __int64 v64; // [rsp+40h] [rbp-100h]
  __int64 v65; // [rsp+40h] [rbp-100h]
  _QWORD *v66; // [rsp+48h] [rbp-F8h]
  __int64 v67; // [rsp+48h] [rbp-F8h]
  __int64 v68; // [rsp+50h] [rbp-F0h]
  int *v69; // [rsp+50h] [rbp-F0h]
  __int64 v70; // [rsp+50h] [rbp-F0h]
  int *v71; // [rsp+50h] [rbp-F0h]
  __int64 v72; // [rsp+58h] [rbp-E8h]
  __int64 v73; // [rsp+58h] [rbp-E8h]
  __int64 v74; // [rsp+58h] [rbp-E8h]
  unsigned int v75; // [rsp+68h] [rbp-D8h] BYREF
  unsigned int v76; // [rsp+6Ch] [rbp-D4h] BYREF
  _BYTE *v77; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v78; // [rsp+78h] [rbp-C8h]
  _BYTE v79[64]; // [rsp+80h] [rbp-C0h] BYREF
  _BYTE *v80; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v81; // [rsp+C8h] [rbp-78h]
  _BYTE v82[24]; // [rsp+D0h] [rbp-70h] BYREF
  int v83; // [rsp+E8h] [rbp-58h] BYREF
  __int64 v84; // [rsp+F0h] [rbp-50h]
  int *v85; // [rsp+F8h] [rbp-48h]
  int *v86; // [rsp+100h] [rbp-40h]
  __int64 v87; // [rsp+108h] [rbp-38h]

  v63 = 0;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 16) + 112LL);
  if ( v5 != sub_1D00B10 )
    v63 = v5();
  v6 = *(_QWORD **)(a1 + 40);
  v83 = 0;
  v84 = 0;
  v66 = v6;
  v77 = v79;
  v78 = 0x800000000LL;
  v80 = v82;
  v81 = 0x400000000LL;
  v85 = &v83;
  v86 = &v83;
  v87 = 0;
  v68 = *(_QWORD *)(a3 + 40);
  if ( *(_QWORD *)(a3 + 32) == v68 )
  {
    v30 = v79;
    v31 = 0;
    goto LABEL_39;
  }
  v7 = *(_QWORD *)(a3 + 32);
  do
  {
    v8 = *(_QWORD *)(*(_QWORD *)v7 + 8LL);
    if ( **(_WORD **)(v8 + 16) )
    {
      if ( **(_WORD **)(v8 + 16) != 45 )
      {
        v9 = *(_QWORD *)(v8 + 32);
        v72 = v9 + 40LL * *(unsigned int *)(v8 + 40);
        if ( v72 != v9 )
        {
          v64 = v7;
          v10 = *(_QWORD *)(v8 + 32);
          while ( 1 )
          {
            while ( *(_BYTE *)v10 || (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
            {
LABEL_20:
              v10 += 40;
              if ( v72 == v10 )
                goto LABEL_21;
            }
            v11 = *(_DWORD *)(v10 + 8);
            v75 = v11;
            if ( (v11 & 0x80000000) == 0 )
            {
              v12 = *(__int64 (**)())(**(_QWORD **)(*v66 + 16LL) + 112LL);
              if ( v12 == sub_1D00B10 )
                BUG();
              a5 = v12();
              if ( *(_BYTE *)(*(_QWORD *)(a5 + 232) + 8LL * v11 + 4) )
              {
                v8 = v11;
                if ( (*(_QWORD *)(v66[38] + 8LL * (v11 >> 6)) & (1LL << v11)) == 0 )
                {
                  if ( !v63 )
                    BUG();
                  v13 = *(_DWORD *)(*(_QWORD *)(v63 + 8) + 24LL * v75 + 16);
                  v14 = v13 & 0xF;
                  LODWORD(a5) = v75 * (v13 & 0xF);
                  v8 = *(_QWORD *)(v63 + 56) + 2LL * (v13 >> 4);
                  LOWORD(a5) = *(_WORD *)v8 + v75 * v14;
                  v15 = v8 + 2;
                  v16 = a5;
LABEL_18:
                  v17 = v15;
                  while ( v17 )
                  {
                    v17 += 2;
                    v76 = (unsigned __int16)v16;
                    sub_1D041C0((__int64)&v80, &v76, v9, v8, a5);
                    v18 = *(unsigned __int16 *)(v17 - 2);
                    v15 = 0;
                    v16 += v18;
                    if ( !(_WORD)v18 )
                      goto LABEL_18;
                  }
                }
              }
              goto LABEL_20;
            }
            v10 += 40;
            sub_1D041C0((__int64)&v80, &v75, v9, v8, a5);
            if ( v72 == v10 )
            {
LABEL_21:
              v7 = v64;
              break;
            }
          }
        }
      }
    }
    v7 += 8;
  }
  while ( v68 != v7 );
  v19 = *(_QWORD *)(a3 + 40);
  if ( v19 != *(_QWORD *)(a3 + 32) )
  {
    v20 = *(_QWORD *)(a3 + 32);
    v21 = v66;
    v22 = &v83;
    while ( 1 )
    {
      v23 = *(_QWORD *)(*(_QWORD *)v20 + 8LL);
      v24 = *(_QWORD *)(v23 + 32);
      v25 = v24 + 40LL * *(unsigned int *)(v23 + 40);
      if ( v24 != v25 )
        break;
LABEL_37:
      v20 += 8;
      if ( v19 == v20 )
        goto LABEL_38;
    }
    while ( 2 )
    {
      if ( *(_BYTE *)v24 )
        goto LABEL_36;
      v26 = *(_BYTE *)(v24 + 3);
      if ( (v26 & 0x10) == 0 || (((v26 & 0x10) != 0) & (v26 >> 6)) != 0 )
        goto LABEL_36;
      v27 = *(unsigned int *)(v24 + 8);
      if ( (int)v27 < 0 )
      {
        if ( !v87 )
        {
          v28 = v80;
          v29 = &v80[4 * (unsigned int)v81];
          if ( v80 != (_BYTE *)v29 )
          {
            while ( (_DWORD)v27 != *v28 )
            {
              if ( v29 == ++v28 )
                goto LABEL_76;
            }
            if ( v28 != v29 )
              goto LABEL_36;
          }
LABEL_76:
          v60 = (unsigned int)v78;
          if ( (unsigned int)v78 >= HIDWORD(v78) )
          {
            v71 = v22;
            v74 = v19;
            sub_16CD150((__int64)&v77, v79, 0, 8, (int)v22, v19);
            v60 = (unsigned int)v78;
            v22 = v71;
            v19 = v74;
          }
          *(_QWORD *)&v77[8 * v60] = v27;
          LODWORD(v78) = v78 + 1;
          goto LABEL_36;
        }
        v56 = v84;
        if ( !v84 )
          goto LABEL_76;
        v57 = v22;
        do
        {
          while ( 1 )
          {
            v58 = *(_QWORD *)(v56 + 16);
            v59 = *(_QWORD *)(v56 + 24);
            if ( (unsigned int)v27 <= *(_DWORD *)(v56 + 32) )
              break;
            v56 = *(_QWORD *)(v56 + 24);
            if ( !v59 )
              goto LABEL_74;
          }
          v57 = (int *)v56;
          v56 = *(_QWORD *)(v56 + 16);
        }
        while ( v58 );
LABEL_74:
        if ( v57 == v22 || (unsigned int)v27 < v57[8] )
          goto LABEL_76;
LABEL_36:
        v24 += 40;
        if ( v25 == v24 )
          goto LABEL_37;
        continue;
      }
      break;
    }
    v69 = v22;
    v73 = v19;
    v32 = *(__int64 (**)())(**(_QWORD **)(*v21 + 16LL) + 112LL);
    if ( v32 == sub_1D00B10 )
      BUG();
    v33 = v32();
    v19 = v73;
    v22 = v69;
    if ( !*(_BYTE *)(*(_QWORD *)(v33 + 232) + 8LL * (unsigned int)v27 + 4)
      || (*(_QWORD *)(v21[38] + 8LL * ((unsigned int)v27 >> 6)) & (1LL << v27)) != 0 )
    {
      goto LABEL_36;
    }
    if ( !v63 )
      BUG();
    v34 = v20;
    v35 = v24;
    v36 = *(_DWORD *)(*(_QWORD *)(v63 + 8) + 24LL * (unsigned int)v27 + 16);
    v37 = v27 * (v36 & 0xF);
    v38 = (_WORD *)(*(_QWORD *)(v63 + 56) + 2LL * (v36 >> 4));
    v39 = v38 + 1;
    v40 = v25;
    v41 = *v38 + v37;
    v42 = v69;
    v43 = v35;
    while ( 2 )
    {
      v44 = v39;
      if ( !v39 )
      {
LABEL_58:
        v48 = v34;
        v49 = v42;
        v19 = v73;
        v25 = v40;
        v24 = v43;
        v22 = v49;
        v20 = v48;
        goto LABEL_36;
      }
LABEL_50:
      if ( v87 )
      {
        v50 = v84;
        if ( v84 )
        {
          v51 = (__int64)v42;
          do
          {
            while ( 1 )
            {
              v52 = *(_QWORD *)(v50 + 16);
              v53 = *(_QWORD *)(v50 + 24);
              if ( (unsigned int)v41 <= *(_DWORD *)(v50 + 32) )
                break;
              v50 = *(_QWORD *)(v50 + 24);
              if ( !v53 )
                goto LABEL_64;
            }
            v51 = v50;
            v50 = *(_QWORD *)(v50 + 16);
          }
          while ( v52 );
LABEL_64:
          if ( (int *)v51 != v42 && (unsigned int)v41 >= *(_DWORD *)(v51 + 32) )
          {
LABEL_56:
            v47 = *v44;
            v39 = 0;
            ++v44;
            if ( !v47 )
              continue;
            v41 += v47;
            if ( !v44 )
              goto LABEL_58;
            goto LABEL_50;
          }
        }
      }
      else
      {
        v45 = v80;
        v46 = &v80[4 * (unsigned int)v81];
        if ( v80 != (_BYTE *)v46 )
        {
          while ( v41 != *v45 )
          {
            if ( v46 == ++v45 )
              goto LABEL_66;
          }
          if ( v46 != v45 )
            goto LABEL_56;
        }
      }
      break;
    }
LABEL_66:
    v54 = (unsigned int)v78;
    v55 = v41;
    if ( (unsigned int)v78 >= HIDWORD(v78) )
    {
      v65 = v43;
      v67 = v34;
      v70 = v40;
      sub_16CD150((__int64)&v77, v79, 0, 8, v40, v34);
      v54 = (unsigned int)v78;
      v43 = v65;
      v34 = v67;
      v40 = v70;
      v55 = v41;
    }
    *(_QWORD *)&v77[8 * v54] = v55;
    LODWORD(v78) = v78 + 1;
    goto LABEL_56;
  }
LABEL_38:
  v30 = v77;
  v31 = (unsigned int)v78;
LABEL_39:
  sub_1EE72E0(a2, v30, v31);
  sub_1E42920(v84);
  if ( v80 != v82 )
    _libc_free((unsigned __int64)v80);
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
}
