// Function: sub_1547D80
// Address: 0x1547d80
//
void __fastcall sub_1547D80(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned int v8; // esi
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 *v11; // r14
  __int64 *v12; // r15
  unsigned __int8 v13; // al
  _BYTE *v14; // rsi
  _BYTE *v15; // r12
  unsigned int v16; // esi
  __int64 v17; // r12
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r13
  __int64 v23; // r14
  __int64 v24; // r12
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 i; // r8
  _BYTE *v29; // r13
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 v32; // r12
  __int64 v33; // rdx
  int v34; // r10d
  __int64 *v35; // r11
  int v36; // ecx
  int v37; // edx
  int v38; // r8d
  int v39; // r8d
  __int64 v40; // r10
  unsigned int v41; // ecx
  __int64 v42; // r9
  int v43; // edi
  __int64 *v44; // rsi
  int v45; // edi
  int v46; // edi
  __int64 v47; // r9
  int v48; // esi
  unsigned int v49; // r13d
  __int64 *v50; // rcx
  __int64 v51; // r8
  __int64 v53; // [rsp+10h] [rbp-B0h]
  __int64 v54; // [rsp+28h] [rbp-98h]
  __int64 v55; // [rsp+30h] [rbp-90h]
  __int64 v56; // [rsp+30h] [rbp-90h]
  __int64 v57; // [rsp+38h] [rbp-88h]
  __int64 v58; // [rsp+38h] [rbp-88h]
  _BYTE *v59; // [rsp+40h] [rbp-80h] BYREF
  __int64 v60; // [rsp+48h] [rbp-78h]
  _BYTE v61[112]; // [rsp+50h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112);
  *(_DWORD *)(a1 + 504) = 0;
  *(_DWORD *)(a1 + 536) = v3 >> 4;
  sub_153EB70(a1, a2);
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2);
    v4 = *(_QWORD *)(a2 + 88);
    v5 = v4 + 40LL * *(_QWORD *)(a2 + 96);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2);
      v4 = *(_QWORD *)(a2 + 88);
    }
  }
  else
  {
    v4 = *(_QWORD *)(a2 + 88);
    v5 = v4 + 40LL * *(_QWORD *)(a2 + 96);
  }
  while ( v5 != v4 )
  {
    v6 = v4;
    v4 += 40;
    sub_15445A0(a1, v6);
  }
  v7 = (__int64)(*(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112)) >> 4;
  *(_DWORD *)(a1 + 548) = v7;
  v8 = v7;
  v54 = a2 + 72;
  v55 = *(_QWORD *)(a2 + 80);
  if ( v55 != a2 + 72 )
  {
    while ( 1 )
    {
      if ( !v55 )
        BUG();
      v9 = *(_QWORD *)(v55 + 24);
      v57 = v55 - 24;
      if ( v9 != v55 + 16 )
      {
        while ( 1 )
        {
          if ( !v9 )
            BUG();
          v10 = 3LL * (*(_DWORD *)(v9 - 4) & 0xFFFFFFF);
          if ( (*(_BYTE *)(v9 - 1) & 0x40) != 0 )
          {
            v11 = *(__int64 **)(v9 - 32);
            v12 = &v11[v10];
          }
          else
          {
            v12 = (__int64 *)(v9 - 24);
            v11 = (__int64 *)(v9 - 24 - v10 * 8);
          }
          if ( v11 != v12 )
            break;
LABEL_19:
          v9 = *(_QWORD *)(v9 + 8);
          if ( v55 + 16 == v9 )
            goto LABEL_20;
        }
        while ( 2 )
        {
          while ( 1 )
          {
            v13 = *(_BYTE *)(*v11 + 16);
            if ( v13 > 0x10u )
              break;
            if ( v13 > 3u )
              goto LABEL_14;
            v11 += 3;
            if ( v12 == v11 )
              goto LABEL_19;
          }
          if ( v13 == 20 )
LABEL_14:
            sub_15445A0(a1, *v11);
          v11 += 3;
          if ( v12 == v11 )
            goto LABEL_19;
          continue;
        }
      }
LABEL_20:
      v14 = *(_BYTE **)(a1 + 520);
      v59 = (_BYTE *)(v55 - 24);
      if ( v14 == *(_BYTE **)(a1 + 528) )
      {
        sub_15409C0(a1 + 512, v14, &v59);
        v15 = *(_BYTE **)(a1 + 520);
      }
      else
      {
        if ( v14 )
        {
          *(_QWORD *)v14 = v57;
          v14 = *(_BYTE **)(a1 + 520);
        }
        v15 = v14 + 8;
        *(_QWORD *)(a1 + 520) = v14 + 8;
      }
      v16 = *(_DWORD *)(a1 + 104);
      v17 = (__int64)&v15[-*(_QWORD *)(a1 + 512)] >> 3;
      if ( !v16 )
        break;
      v18 = *(_QWORD *)(a1 + 88);
      LODWORD(v19) = (v16 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
      v20 = (__int64 *)(v18 + 16LL * (unsigned int)v19);
      v21 = *v20;
      if ( v57 != *v20 )
      {
        v34 = 1;
        v35 = 0;
        while ( v21 != -8 )
        {
          if ( v21 == -16 && !v35 )
            v35 = v20;
          v19 = (v16 - 1) & ((_DWORD)v19 + v34);
          v20 = (__int64 *)(v18 + 16 * v19);
          v21 = *v20;
          if ( v57 == *v20 )
            goto LABEL_26;
          ++v34;
        }
        v36 = *(_DWORD *)(a1 + 96);
        if ( v35 )
          v20 = v35;
        ++*(_QWORD *)(a1 + 80);
        v37 = v36 + 1;
        if ( 4 * (v36 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(a1 + 100) - v37 <= v16 >> 3 )
          {
            sub_1542080(a1 + 80, v16);
            v45 = *(_DWORD *)(a1 + 104);
            if ( !v45 )
            {
LABEL_96:
              ++*(_DWORD *)(a1 + 96);
              BUG();
            }
            v46 = v45 - 1;
            v47 = *(_QWORD *)(a1 + 88);
            v48 = 1;
            v49 = v46 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
            v37 = *(_DWORD *)(a1 + 96) + 1;
            v50 = 0;
            v20 = (__int64 *)(v47 + 16LL * v49);
            v51 = *v20;
            if ( v57 != *v20 )
            {
              while ( v51 != -8 )
              {
                if ( !v50 && v51 == -16 )
                  v50 = v20;
                v49 = v46 & (v48 + v49);
                v20 = (__int64 *)(v47 + 16LL * v49);
                v51 = *v20;
                if ( v57 == *v20 )
                  goto LABEL_62;
                ++v48;
              }
              if ( v50 )
                v20 = v50;
            }
          }
          goto LABEL_62;
        }
LABEL_66:
        sub_1542080(a1 + 80, 2 * v16);
        v38 = *(_DWORD *)(a1 + 104);
        if ( !v38 )
          goto LABEL_96;
        v39 = v38 - 1;
        v40 = *(_QWORD *)(a1 + 88);
        v37 = *(_DWORD *)(a1 + 96) + 1;
        v41 = v39 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v20 = (__int64 *)(v40 + 16LL * v41);
        v42 = *v20;
        if ( v57 != *v20 )
        {
          v43 = 1;
          v44 = 0;
          while ( v42 != -8 )
          {
            if ( !v44 && v42 == -16 )
              v44 = v20;
            v41 = v39 & (v43 + v41);
            v20 = (__int64 *)(v40 + 16LL * v41);
            v42 = *v20;
            if ( v57 == *v20 )
              goto LABEL_62;
            ++v43;
          }
          if ( v44 )
            v20 = v44;
        }
LABEL_62:
        *(_DWORD *)(a1 + 96) = v37;
        if ( *v20 != -8 )
          --*(_DWORD *)(a1 + 100);
        *((_DWORD *)v20 + 2) = 0;
        *v20 = v57;
      }
LABEL_26:
      *((_DWORD *)v20 + 2) = v17;
      v55 = *(_QWORD *)(v55 + 8);
      if ( v54 == v55 )
      {
        v8 = *(_DWORD *)(a1 + 548);
        v7 = (__int64)(*(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112)) >> 4;
        goto LABEL_28;
      }
    }
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_66;
  }
LABEL_28:
  sub_1542240(a1, v8, v7);
  sub_1546040(a1, *(_QWORD *)(a2 + 112));
  *(_DWORD *)(a1 + 552) = (__int64)(*(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112)) >> 4;
  v22 = *(_QWORD *)(a2 + 80);
  v59 = v61;
  v60 = 0x800000000LL;
  if ( v54 != v22 )
  {
    do
    {
      if ( !v22 )
        BUG();
      v23 = *(_QWORD *)(v22 + 24);
      v24 = v22 + 16;
      if ( v22 + 16 != v23 )
      {
        v53 = v22;
        do
        {
          if ( !v23 )
            BUG();
          v25 = 24LL * (*(_DWORD *)(v23 - 4) & 0xFFFFFFF);
          if ( (*(_BYTE *)(v23 - 1) & 0x40) != 0 )
          {
            v26 = *(_QWORD *)(v23 - 32);
            v27 = v26 + v25;
          }
          else
          {
            v27 = v23 - 24;
            v26 = v23 - 24 - v25;
          }
          for ( i = v27; i != v26; v26 += 24 )
          {
            if ( *(_BYTE *)(*(_QWORD *)v26 + 16LL) == 19 )
            {
              v29 = *(_BYTE **)(*(_QWORD *)v26 + 24LL);
              if ( *v29 == 2 )
              {
                v30 = (unsigned int)v60;
                if ( (unsigned int)v60 >= HIDWORD(v60) )
                {
                  v56 = i;
                  v58 = v26;
                  sub_16CD150(&v59, v61, 0, 8);
                  v30 = (unsigned int)v60;
                  i = v56;
                  v26 = v58;
                }
                *(_QWORD *)&v59[8 * v30] = v29;
                LODWORD(v60) = v60 + 1;
              }
            }
          }
          if ( *(_BYTE *)(*(_QWORD *)(v23 - 24) + 8LL) )
            sub_15445A0(a1, v23 - 24);
          v23 = *(_QWORD *)(v23 + 8);
        }
        while ( v24 != v23 );
        v22 = v53;
      }
      v22 = *(_QWORD *)(v22 + 8);
    }
    while ( v54 != v22 );
    if ( (_DWORD)v60 )
    {
      v31 = 8LL * (unsigned int)v60;
      v32 = 0;
      do
      {
        v33 = *(_QWORD *)&v59[v32];
        v32 += 8;
        sub_1545BA0(a1, a2, v33);
      }
      while ( v31 != v32 );
    }
    if ( v59 != v61 )
      _libc_free((unsigned __int64)v59);
  }
}
