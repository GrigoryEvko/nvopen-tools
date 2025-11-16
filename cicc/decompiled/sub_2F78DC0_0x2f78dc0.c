// Function: sub_2F78DC0
// Address: 0x2f78dc0
//
void __fastcall sub_2F78DC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // r8
  int *v6; // r12
  int v7; // ebx
  unsigned __int128 v8; // rax
  unsigned __int128 v9; // kr00_16
  unsigned __int64 v10; // r12
  _BYTE *v11; // rbx
  unsigned int v12; // esi
  unsigned int v13; // edi
  __int64 v14; // rcx
  unsigned int v15; // eax
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // r10
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // rax
  _QWORD *v23; // r13
  __int64 v24; // rax
  char v25; // dl
  __int64 v26; // rcx
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // r10
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int64 v34; // r13
  __int64 v35; // r11
  unsigned __int64 v36; // rdi
  __int64 v37; // rsi
  unsigned __int64 j; // rdx
  __int64 k; // r8
  __int16 v40; // cx
  __int64 v41; // rdi
  __int64 v42; // r8
  unsigned int v43; // esi
  __int64 *v44; // rcx
  __int64 v45; // r9
  unsigned int v46; // edx
  _QWORD *v47; // rdx
  unsigned __int64 v48; // r15
  int v49; // ecx
  __int64 v50; // rcx
  __int64 v51; // r9
  unsigned __int64 v52; // rax
  __int64 i; // rdi
  __int16 v54; // dx
  unsigned int v55; // edi
  __int64 v56; // r8
  unsigned int v57; // ecx
  __int64 *v58; // rdx
  __int64 v59; // r10
  int v60; // edx
  int v61; // r11d
  __int64 v62; // [rsp+8h] [rbp-2F8h]
  __int64 v63; // [rsp+10h] [rbp-2F0h]
  int v64; // [rsp+10h] [rbp-2F0h]
  int *v65; // [rsp+30h] [rbp-2D0h]
  unsigned int v66; // [rsp+40h] [rbp-2C0h]
  signed __int64 v67; // [rsp+48h] [rbp-2B8h]
  unsigned __int64 v68; // [rsp+50h] [rbp-2B0h]
  __int64 v69; // [rsp+58h] [rbp-2A8h]
  int *v70; // [rsp+60h] [rbp-2A0h] BYREF
  __int64 v71; // [rsp+68h] [rbp-298h]
  _BYTE v72[192]; // [rsp+70h] [rbp-290h] BYREF
  _BYTE *v73; // [rsp+130h] [rbp-1D0h]
  __int64 v74; // [rsp+138h] [rbp-1C8h]
  _BYTE v75[192]; // [rsp+140h] [rbp-1C0h] BYREF
  unsigned int *v76; // [rsp+200h] [rbp-100h]
  __int64 v77; // [rsp+208h] [rbp-F8h]
  _BYTE v78[240]; // [rsp+210h] [rbp-F0h] BYREF

  v67 = 0;
  if ( *(_BYTE *)(a1 + 56) )
  {
    v50 = a2;
    v51 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
    v52 = a2;
    if ( (*(_DWORD *)(a2 + 44) & 4) != 0 )
    {
      do
        v52 = *(_QWORD *)v52 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v52 + 44) & 4) != 0 );
    }
    if ( (*(_DWORD *)(a2 + 44) & 8) != 0 )
    {
      do
        v50 = *(_QWORD *)(v50 + 8);
      while ( (*(_BYTE *)(v50 + 44) & 8) != 0 );
    }
    for ( i = *(_QWORD *)(v50 + 8); i != v52; v52 = *(_QWORD *)(v52 + 8) )
    {
      v54 = *(_WORD *)(v52 + 68);
      if ( (unsigned __int16)(v54 - 14) > 4u && v54 != 24 )
        break;
    }
    v55 = *(_DWORD *)(v51 + 144);
    v56 = *(_QWORD *)(v51 + 128);
    if ( v55 )
    {
      v57 = (v55 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
      v58 = (__int64 *)(v56 + 16LL * v57);
      v59 = *v58;
      if ( *v58 == v52 )
      {
LABEL_77:
        v67 = v58[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
        goto LABEL_2;
      }
      v60 = 1;
      while ( v59 != -4096 )
      {
        v61 = v60 + 1;
        v57 = (v55 - 1) & (v60 + v57);
        v58 = (__int64 *)(v56 + 16LL * v57);
        v59 = *v58;
        if ( *v58 == v52 )
          goto LABEL_77;
        v60 = v61;
      }
    }
    v58 = (__int64 *)(v56 + 16LL * v55);
    goto LABEL_77;
  }
LABEL_2:
  v3 = *(_QWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned __int8 *)(a1 + 58);
  v70 = (int *)v72;
  v73 = v75;
  v71 = 0x800000000LL;
  v74 = 0x800000000LL;
  v76 = (unsigned int *)v78;
  v77 = 0x800000000LL;
  sub_2F75980((__int64)&v70, a2, v4, v3, v5, 0);
  if ( *(_BYTE *)(a1 + 58) )
    sub_2F76630((__int64 *)&v70, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24), v67, 0);
  if ( *(_BYTE *)(a1 + 56) )
  {
    v65 = &v70[6 * (unsigned int)v71];
    if ( v65 != v70 )
    {
      v6 = v70;
      do
      {
        v7 = *v6;
        v66 = *v6;
        *(_QWORD *)&v8 = sub_2F77D00(a1, *v6, v67);
        v9 = v8;
        if ( v8 != 0 )
        {
          v22 = sub_2F75400((_QWORD *)a1);
          v23 = *(_QWORD **)(a1 + 24);
          v63 = v22;
          v69 = *(_QWORD *)(a1 + 32);
          v62 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v23 + 16LL) + 200LL))(*(_QWORD *)(*v23 + 16LL));
          if ( v7 < 0 )
            v24 = *(_QWORD *)(v23[7] + 16LL * (v7 & 0x7FFFFFFF) + 8);
          else
            v24 = *(_QWORD *)(v23[38] + 8LL * (unsigned int)v7);
          while ( v24 )
          {
            if ( (*(_BYTE *)(v24 + 3) & 0x10) == 0 )
            {
              v25 = *(_BYTE *)(v24 + 4);
              if ( (v25 & 8) == 0 )
              {
                v34 = v63 & 0xFFFFFFFFFFFFFFF8LL;
                v35 = (v63 >> 1) & 3;
LABEL_44:
                if ( (v25 & 1) != 0 )
                  goto LABEL_57;
                v36 = *(_QWORD *)(v24 + 16);
                v37 = *(_QWORD *)(v69 + 32);
                for ( j = v36; (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
                  ;
                for ( ; (*(_BYTE *)(v36 + 44) & 8) != 0; v36 = *(_QWORD *)(v36 + 8) )
                  ;
                for ( k = *(_QWORD *)(v36 + 8); k != j; j = *(_QWORD *)(j + 8) )
                {
                  v40 = *(_WORD *)(j + 68);
                  if ( (unsigned __int16)(v40 - 14) > 4u && v40 != 24 )
                    break;
                }
                v41 = *(unsigned int *)(v37 + 144);
                v42 = *(_QWORD *)(v37 + 128);
                if ( (_DWORD)v41 )
                {
                  v43 = (v41 - 1) & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
                  v44 = (__int64 *)(v42 + 16LL * v43);
                  v45 = *v44;
                  if ( j == *v44 )
                    goto LABEL_55;
                  v49 = 1;
                  while ( v45 != -4096 )
                  {
                    v43 = (v41 - 1) & (v49 + v43);
                    v64 = v49 + 1;
                    v44 = (__int64 *)(v42 + 16LL * v43);
                    v45 = *v44;
                    if ( *v44 == j )
                      goto LABEL_55;
                    v49 = v64;
                  }
                }
                v44 = (__int64 *)(v42 + 16 * v41);
LABEL_55:
                v46 = *(_DWORD *)((v44[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | 2;
                if ( v46 >= ((unsigned int)v35 | *(_DWORD *)(v34 + 24))
                  && v46 < (*(_DWORD *)((v67 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v67 >> 1) & 2) )
                {
                  v47 = (_QWORD *)(*(_QWORD *)(v62 + 272) + 16LL * ((*(_DWORD *)v24 >> 8) & 0xFFF));
                  v68 = ~v47[1] & *((_QWORD *)&v9 + 1);
                  v48 = ~*v47 & v9;
                  v9 = __PAIR128__(v68, v48);
                  if ( !(v48 | v68) )
                    goto LABEL_8;
                }
LABEL_57:
                while ( 1 )
                {
                  v24 = *(_QWORD *)(v24 + 32);
                  if ( !v24 )
                    goto LABEL_33;
                  if ( (*(_BYTE *)(v24 + 3) & 0x10) == 0 )
                  {
                    v25 = *(_BYTE *)(v24 + 4);
                    if ( (v25 & 8) == 0 )
                      goto LABEL_44;
                  }
                }
              }
            }
            v24 = *(_QWORD *)(v24 + 32);
          }
LABEL_33:
          if ( v9 != 0 )
          {
            if ( v7 < 0 )
              v7 = *(_DWORD *)(a1 + 320) + (v7 & 0x7FFFFFFF);
            v26 = *(unsigned int *)(a1 + 104);
            v27 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 304) + (unsigned int)v7);
            if ( v27 >= (unsigned int)v26 )
              goto LABEL_78;
            v28 = *(_QWORD *)(a1 + 96);
            while ( 1 )
            {
              v29 = v28 + 24LL * v27;
              if ( v7 == *(_DWORD *)v29 )
                break;
              v27 += 256;
              if ( (unsigned int)v26 <= v27 )
                goto LABEL_78;
            }
            if ( v29 == v28 + 24 * v26 )
            {
LABEL_78:
              v32 = 0;
              v33 = 0;
              v31 = 0;
              v30 = 0;
            }
            else
            {
              v30 = *(_QWORD *)(v29 + 8);
              v31 = *(_QWORD *)(v29 + 16);
              v32 = v30 & ~(_QWORD)v9;
              v33 = v31 & ~*((_QWORD *)&v9 + 1);
            }
            sub_2F74F40(a1, v66, v30, v31, v32, v33);
          }
        }
LABEL_8:
        v6 += 6;
      }
      while ( v65 != v6 );
    }
  }
  v10 = (unsigned __int64)v73;
  v11 = &v73[24 * (unsigned int)v74];
  if ( v11 != v73 )
  {
    do
    {
      v12 = *(_DWORD *)v10;
      v13 = *(_DWORD *)v10;
      if ( *(int *)v10 < 0 )
        v13 = *(_DWORD *)(a1 + 320) + (v13 & 0x7FFFFFFF);
      v14 = *(unsigned int *)(a1 + 104);
      v15 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 304) + v13);
      if ( v15 >= (unsigned int)v14 )
        goto LABEL_26;
      v16 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v17 = v16 + 24LL * v15;
        if ( v13 == *(_DWORD *)v17 )
          break;
        v15 += 256;
        if ( (unsigned int)v14 <= v15 )
          goto LABEL_26;
      }
      if ( v17 == v16 + 24 * v14 )
      {
LABEL_26:
        v19 = 0;
        v18 = 0;
      }
      else
      {
        v18 = *(_QWORD *)(v17 + 8);
        v19 = *(_QWORD *)(v17 + 16);
      }
      v20 = *(_QWORD *)(v10 + 8);
      v21 = *(_QWORD *)(v10 + 16);
      v10 += 24LL;
      sub_2F74DB0(a1, v12, v18, v19, v18 | v20, v19 | v21);
    }
    while ( v11 != (_BYTE *)v10 );
  }
  sub_2F77060(a1, v76, (unsigned int)v77);
  if ( v76 != (unsigned int *)v78 )
    _libc_free((unsigned __int64)v76);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  if ( v70 != (int *)v72 )
    _libc_free((unsigned __int64)v70);
}
