// Function: sub_1EDA110
// Address: 0x1eda110
//
__int64 __fastcall sub_1EDA110(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // r10
  __int64 v5; // rdx
  __int64 v7; // rbx
  __int64 v8; // rdx
  int v9; // r9d
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // rax
  int v13; // r9d
  __int64 v14; // r10
  __int64 v15; // rbx
  unsigned __int64 v16; // rax
  unsigned int v17; // r13d
  int v18; // ecx
  __int64 v19; // r14
  __int64 *v20; // rdx
  __int64 *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // r11
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // r11
  int v31; // ecx
  __int64 v32; // r10
  int v33; // r15d
  __int64 v34; // rbx
  __int64 v35; // r13
  unsigned int v36; // esi
  int v37; // r14d
  __int64 v38; // r12
  char v39; // al
  _QWORD *v40; // rdi
  __int64 v41; // rdx
  unsigned int v42; // eax
  __int64 v43; // rax
  _BYTE *v44; // rax
  unsigned __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // [rsp+8h] [rbp-108h]
  __int64 v48; // [rsp+8h] [rbp-108h]
  __int64 v49; // [rsp+18h] [rbp-F8h]
  __int64 v50; // [rsp+20h] [rbp-F0h]
  __int64 v51; // [rsp+28h] [rbp-E8h]
  __int64 v52; // [rsp+28h] [rbp-E8h]
  int v53; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v54; // [rsp+30h] [rbp-E0h]
  int v55; // [rsp+30h] [rbp-E0h]
  __int64 v56; // [rsp+38h] [rbp-D8h]
  int v57; // [rsp+38h] [rbp-D8h]
  __int64 v58; // [rsp+38h] [rbp-D8h]
  __int64 v59; // [rsp+38h] [rbp-D8h]
  __int64 v60; // [rsp+40h] [rbp-D0h]
  int v61; // [rsp+40h] [rbp-D0h]
  __int64 v62; // [rsp+40h] [rbp-D0h]
  __int64 v63; // [rsp+48h] [rbp-C8h]
  __int64 v64; // [rsp+48h] [rbp-C8h]
  __int64 v65; // [rsp+48h] [rbp-C8h]
  _BYTE *v66; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+58h] [rbp-B8h]
  _BYTE v68[176]; // [rsp+60h] [rbp-B0h] BYREF

  v2 = a1;
  v3 = 0;
  v5 = *(unsigned int *)(*(_QWORD *)a1 + 72LL);
  v49 = 8 * v5;
  if ( !(_DWORD)v5 )
    return 1;
LABEL_2:
  v50 = *(_QWORD *)(v2 + 112) + 5 * v3;
  if ( *(_DWORD *)v50 != 4 )
    goto LABEL_3;
  if ( *(_BYTE *)(v2 + 20) )
    return 0;
  v56 = v3;
  v7 = *(_QWORD *)(v2 + 48);
  v60 = v2;
  v8 = *(_QWORD *)(*(_QWORD *)v2 + 64LL);
  v63 = *(_QWORD *)(v8 + v3);
  v9 = *(_DWORD *)(v50 + 4) & *(_DWORD *)(*(_QWORD *)(a2 + 112) + 40LL * **(unsigned int **)(v50 + 24) + 8);
  v66 = v68;
  v67 = 0x800000000LL;
  v53 = v9;
  v10 = *(_QWORD *)(*(_QWORD *)(v8 + v3) + 8LL);
  v11 = *(_QWORD *)(*(_QWORD *)(v7 + 392) + 16LL * *(unsigned int *)(sub_1DA9310(v7, v10) + 48) + 8);
  v12 = sub_1DB3C70(*(__int64 **)a2, v10);
  v13 = v53;
  v14 = v56;
  v15 = v12;
  v2 = v60;
  v16 = v11 & 0xFFFFFFFFFFFFFFF8LL;
  v17 = v53;
  v18 = *(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  v19 = (v11 >> 1) & 3;
  while ( 1 )
  {
    v23 = *(_QWORD *)(v15 + 8);
    v24 = (v23 >> 1) & 3;
    if ( ((unsigned int)v24 | *(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24)) >= ((unsigned int)v19 | v18) )
      break;
    if ( v24 != 3 )
    {
      v25 = (unsigned int)v67;
      v26 = v17;
      if ( (unsigned int)v67 >= HIDWORD(v67) )
      {
        v47 = v2;
        v51 = v14;
        v54 = v16;
        v57 = v13;
        sub_16CD150((__int64)&v66, v68, 0, 16, v2, v13);
        v25 = (unsigned int)v67;
        v2 = v47;
        v14 = v51;
        v16 = v54;
        v13 = v57;
        v26 = v17;
      }
      v20 = (__int64 *)&v66[16 * v25];
      v15 += 24;
      *v20 = v23;
      v20[1] = v26;
      v21 = *(__int64 **)a2;
      LODWORD(v67) = v67 + 1;
      if ( v15 != *v21 + 24LL * *((unsigned int *)v21 + 2) )
      {
        v18 = *(_DWORD *)(v16 + 24);
        if ( (*(_DWORD *)((*(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)v15 >> 1) & 3) < (v18 | (unsigned int)v19) )
        {
          v22 = *(_QWORD *)(a2 + 112) + 40LL * **(unsigned int **)(v15 + 16);
          v17 &= ~*(_DWORD *)(v22 + 4);
          if ( *(_QWORD *)(v22 + 16) )
          {
            if ( v17 )
              continue;
          }
        }
      }
    }
    v27 = *(_QWORD *)(v63 + 8);
    if ( (v27 & 6) != 0 )
    {
      v45 = v27 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v45 || (v46 = *(_QWORD *)(v45 + 16)) == 0 )
        BUG();
      if ( (*(_BYTE *)v46 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v46 + 46) & 8) != 0 )
          v46 = *(_QWORD *)(v46 + 8);
      }
      v29 = *(_QWORD *)(v46 + 8);
    }
    else
    {
      v58 = v14;
      v61 = v13;
      v64 = v2;
      v28 = sub_1DA9310(*(_QWORD *)(v2 + 48), v27);
      v2 = v64;
      v13 = v61;
      v29 = *(_QWORD *)(v28 + 32);
      v14 = v58;
    }
    v30 = 0;
    if ( (*(_QWORD *)v66 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      v30 = *(_QWORD *)((*(_QWORD *)v66 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    v48 = v14;
    v31 = 0;
    v32 = a2;
    v33 = v13;
    while ( 2 )
    {
      if ( (unsigned __int16)(**(_WORD **)(v29 + 16) - 12) > 1u )
      {
        v34 = *(_QWORD *)(v29 + 32);
        v35 = v34 + 40LL * *(unsigned int *)(v29 + 40);
        if ( v34 != v35 )
        {
          v36 = *(_DWORD *)(v32 + 12);
          v37 = *(_DWORD *)(v32 + 8);
          v65 = v29;
          v38 = v2;
          while ( 1 )
          {
            if ( !*(_BYTE *)v34 && (*(_BYTE *)(v34 + 3) & 0x10) == 0 && v37 == *(_DWORD *)(v34 + 8) )
            {
              v39 = *(_BYTE *)(v34 + 4);
              if ( (v39 & 1) == 0 && (v39 & 2) == 0 )
              {
                v40 = *(_QWORD **)(v38 + 56);
                v41 = (*(_DWORD *)v34 >> 8) & 0xFFF;
                if ( v36 )
                {
                  if ( !(_DWORD)v41 )
                  {
                    if ( (*(_DWORD *)(v40[31] + 4LL * v36) & v33) != 0 )
                      goto LABEL_45;
                    goto LABEL_32;
                  }
                  v52 = v32;
                  v55 = v31;
                  v59 = v30;
                  v42 = (*(__int64 (__fastcall **)(_QWORD *))(*v40 + 120LL))(v40);
                  v40 = *(_QWORD **)(v38 + 56);
                  v30 = v59;
                  v31 = v55;
                  v41 = v42;
                  v32 = v52;
                }
                if ( (*(_DWORD *)(v40[31] + 4 * v41) & v33) != 0 )
                  goto LABEL_45;
              }
            }
LABEL_32:
            v34 += 40;
            if ( v35 == v34 )
            {
              v2 = v38;
              v29 = v65;
              break;
            }
          }
        }
      }
      if ( v30 != v29 )
        goto LABEL_35;
      v43 = (unsigned int)(v31 + 1);
      v31 = v43;
      if ( (_DWORD)v43 != (_DWORD)v67 )
      {
        v44 = &v66[16 * v43];
        v30 = 0;
        if ( (*(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          v30 = *(_QWORD *)((*(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL) + 16);
        v33 = *((_DWORD *)v44 + 2);
LABEL_35:
        if ( (*(_BYTE *)v29 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v29 + 46) & 8) != 0 )
            v29 = *(_QWORD *)(v29 + 8);
        }
        v29 = *(_QWORD *)(v29 + 8);
        continue;
      }
      break;
    }
    a2 = v32;
    v3 = v48;
    *(_DWORD *)v50 = 3;
    if ( v66 != v68 )
    {
      v62 = v2;
      _libc_free((unsigned __int64)v66);
      v3 = v48;
      v2 = v62;
    }
LABEL_3:
    v3 += 8;
    if ( v49 == v3 )
      return 1;
    goto LABEL_2;
  }
LABEL_45:
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  return 0;
}
