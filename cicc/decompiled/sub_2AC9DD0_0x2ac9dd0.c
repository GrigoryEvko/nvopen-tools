// Function: sub_2AC9DD0
// Address: 0x2ac9dd0
//
__int64 __fastcall sub_2AC9DD0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r15
  unsigned __int8 *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rdi
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int64 v15; // rax
  __int64 v16; // r9
  _QWORD *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rdx
  __int64 *v29; // rbx
  __int64 v30; // r13
  __int64 v31; // r15
  __int64 v32; // r14
  __int64 v33; // rax
  char v34; // dh
  __int64 v35; // rcx
  __int64 v36; // rsi
  char v37; // al
  __int64 v38; // r8
  __int64 v39; // r10
  __int64 v40; // r9
  __int64 v41; // rdx
  __int64 v42; // rdi
  __int64 v43; // rsi
  __int64 *v44; // rax
  unsigned int v45; // r8d
  __int64 v46; // rdi
  __int64 v47; // r9
  _QWORD *v48; // rsi
  _QWORD *v49; // r9
  __int64 v50; // rdx
  __int64 v51; // rdi
  __int64 v52; // r10
  _QWORD *v53; // rax
  _QWORD *v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rax
  unsigned int v57; // r9d
  __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // rax
  unsigned int v61; // r8d
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // r12
  __int64 v66; // rdx
  const char *v68; // [rsp+18h] [rbp-68h]
  __int64 v69; // [rsp+18h] [rbp-68h]
  const char *v70; // [rsp+20h] [rbp-60h] BYREF
  __int64 v71; // [rsp+28h] [rbp-58h]
  _BYTE v72[16]; // [rsp+30h] [rbp-50h] BYREF
  char v73; // [rsp+40h] [rbp-40h]
  char v74; // [rsp+41h] [rbp-3Fh]

  v3 = (__int64)a1;
  sub_2AB3250(a1, "vec.epilog.", (void *)0xB);
  v4 = (unsigned __int8 *)a1[30];
  v74 = 1;
  v73 = 3;
  v70 = "vec.epilog.ph";
  sub_BD6B50(v4, &v70);
  v5 = *(_QWORD *)(v3 + 32);
  v74 = 1;
  v6 = *(_QWORD *)(v3 + 240);
  v73 = 3;
  v70 = "vec.epilog.iter.check";
  v7 = sub_F36960(v6, *(__int64 **)(v6 + 56), 1, v5, *(_QWORD *)(v3 + 24), 0, (void **)&v70, 1);
  sub_2ABA4F0(v3, *(_QWORD *)(v3 + 248), v7);
  v8 = *(_QWORD *)(v3 + 480);
  *(_QWORD *)(v3 + 456) = v7;
  v9 = sub_986580(*(_QWORD *)(v8 + 24));
  sub_BD2ED0(v9, v7, *(_QWORD *)(v3 + 240));
  v10 = sub_986580(*(_QWORD *)(*(_QWORD *)(v3 + 480) + 32LL));
  sub_BD2ED0(v10, v7, *(_QWORD *)(v3 + 248));
  v11 = *(__int64 **)(v3 + 480);
  v12 = v11[5];
  if ( v12 )
  {
    v13 = sub_986580(v12);
    sub_BD2ED0(v13, v7, *(_QWORD *)(v3 + 248));
    v11 = *(__int64 **)(v3 + 480);
  }
  v14 = v11[6];
  if ( v14 )
  {
    v15 = sub_986580(v14);
    sub_BD2ED0(v15, v7, *(_QWORD *)(v3 + 248));
    v11 = *(__int64 **)(v3 + 480);
  }
  sub_B1AEF0(*(_QWORD *)(v3 + 32), *(_QWORD *)(v3 + 248), v11[4]);
  v17 = *(_QWORD **)(v3 + 480);
  v18 = v17[5];
  if ( v18 )
  {
    sub_B1A4E0(v3 + 264, v18);
    v17 = *(_QWORD **)(v3 + 480);
  }
  v19 = v17[6];
  if ( v19 )
  {
    sub_B1A4E0(v3 + 264, v19);
    v17 = *(_QWORD **)(v3 + 480);
  }
  v20 = v17[4];
  v21 = *(unsigned int *)(v3 + 272);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(v3 + 276) )
  {
    v69 = v20;
    sub_C8D5F0(v3 + 264, (const void *)(v3 + 280), v21 + 1, 8u, v20, v16);
    v21 = *(unsigned int *)(v3 + 272);
    v20 = v69;
  }
  *(_QWORD *)(*(_QWORD *)(v3 + 264) + 8 * v21) = v20;
  ++*(_DWORD *)(v3 + 272);
  v70 = v72;
  v71 = 0x400000000LL;
  v25 = sub_AA5930(v7);
  v26 = (unsigned int)v71;
  if ( v22 != v25 )
  {
    v27 = v22;
    do
    {
      if ( v26 + 1 > (unsigned __int64)HIDWORD(v71) )
      {
        sub_C8D5F0((__int64)&v70, v72, v26 + 1, 8u, v23, v24);
        v26 = (unsigned int)v71;
      }
      *(_QWORD *)&v70[8 * v26] = v25;
      v26 = (unsigned int)(v71 + 1);
      LODWORD(v71) = v71 + 1;
      if ( !v25 )
        BUG();
      v28 = *(_QWORD *)(v25 + 32);
      if ( !v28 )
        BUG();
      v25 = 0;
      if ( *(_BYTE *)(v28 - 24) == 84 )
        v25 = v28 - 24;
    }
    while ( v27 != v25 );
  }
  v29 = (__int64 *)v70;
  v68 = &v70[8 * v26];
  if ( v68 != v70 )
  {
    v30 = v3;
    v31 = v2;
    while ( 1 )
    {
      v32 = *v29;
      LOBYTE(v31) = 1;
      v33 = sub_AA4FF0(*(_QWORD *)(v30 + 240));
      v35 = v31;
      v36 = v33;
      v37 = 0;
      if ( v36 )
        v37 = v34;
      BYTE1(v35) = v37;
      v31 = v35;
      sub_B444E0((_QWORD *)v32, v36, v35);
      v38 = sub_AA54C0(v7);
      if ( (*(_DWORD *)(v32 + 4) & 0x7FFFFFF) == 0 )
        goto LABEL_53;
      v39 = *(_QWORD *)(v32 - 8);
      v40 = *(unsigned int *)(v32 + 72);
      v41 = 0;
      v42 = 8LL * (*(_DWORD *)(v32 + 4) & 0x7FFFFFF);
      do
      {
        while ( 1 )
        {
          v43 = 32 * v40;
          v44 = (__int64 *)(v39 + 32 * v40 + v41);
          if ( v38 == *v44 )
            break;
          v41 += 8;
          if ( v42 == v41 )
            goto LABEL_29;
        }
        *v44 = v7;
        v40 = *(unsigned int *)(v32 + 72);
        v41 += 8;
        v39 = *(_QWORD *)(v32 - 8);
        v43 = 32 * v40;
      }
      while ( v42 != v41 );
LABEL_29:
      v45 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
      v46 = 8LL * v45;
      v47 = v46 + v43;
      v48 = (_QWORD *)(v39 + v43);
      v49 = (_QWORD *)(v39 + v47);
      v50 = v46 >> 3;
      v51 = v46 >> 5;
      if ( v51 )
      {
        v52 = *(_QWORD *)(v30 + 480);
        v53 = v48;
        v54 = &v48[4 * v51];
        v55 = *(_QWORD *)(v52 + 32);
        while ( *v53 != v55 )
        {
          if ( v55 == v53[1] )
          {
            ++v53;
            break;
          }
          if ( v55 == v53[2] )
          {
            v53 += 2;
            break;
          }
          if ( v55 == v53[3] )
          {
            v53 += 3;
            break;
          }
          v53 += 4;
          if ( v54 == v53 )
          {
            v50 = v49 - v53;
            goto LABEL_63;
          }
        }
LABEL_36:
        if ( v53 != v49 )
        {
          if ( v45 )
          {
            v56 = 0;
            while ( 1 )
            {
              v57 = v56;
              if ( *(_QWORD *)(v52 + 32) == v48[v56] )
                break;
              if ( v45 == (_DWORD)++v56 )
                goto LABEL_58;
            }
          }
          else
          {
LABEL_58:
            v57 = -1;
          }
          sub_B48BF0(v32, v57, 1);
          v58 = *(_QWORD *)(v30 + 480);
          v59 = *(_QWORD *)(v58 + 40);
          if ( v59 )
          {
            if ( (*(_DWORD *)(v32 + 4) & 0x7FFFFFF) != 0 )
            {
              v60 = 0;
              while ( 1 )
              {
                v61 = v60;
                if ( v59 == *(_QWORD *)(*(_QWORD *)(v32 - 8) + 32LL * *(unsigned int *)(v32 + 72) + 8 * v60) )
                  break;
                if ( (*(_DWORD *)(v32 + 4) & 0x7FFFFFF) == (_DWORD)++v60 )
                  goto LABEL_74;
              }
            }
            else
            {
LABEL_74:
              v61 = -1;
            }
            sub_B48BF0(v32, v61, 1);
            v58 = *(_QWORD *)(v30 + 480);
          }
          v62 = *(_QWORD *)(v58 + 48);
          if ( v62 )
          {
            if ( (*(_DWORD *)(v32 + 4) & 0x7FFFFFF) != 0 )
            {
              v63 = 0;
              while ( v62 != *(_QWORD *)(*(_QWORD *)(v32 - 8) + 32LL * *(unsigned int *)(v32 + 72) + 8 * v63) )
              {
                if ( (*(_DWORD *)(v32 + 4) & 0x7FFFFFF) == (_DWORD)++v63 )
                  goto LABEL_73;
              }
              sub_B48BF0(v32, v63, 1);
            }
            else
            {
LABEL_73:
              sub_B48BF0(v32, 0xFFFFFFFF, 1);
            }
          }
        }
        goto LABEL_53;
      }
      v53 = v48;
LABEL_63:
      switch ( v50 )
      {
        case 2LL:
          v52 = *(_QWORD *)(v30 + 480);
          v66 = *(_QWORD *)(v52 + 32);
LABEL_65:
          if ( *v53 == v66 )
            goto LABEL_36;
          ++v53;
LABEL_70:
          if ( *v53 == v66 )
            goto LABEL_36;
          if ( v68 == (const char *)++v29 )
          {
LABEL_54:
            v3 = v30;
            goto LABEL_55;
          }
          break;
        case 3LL:
          v52 = *(_QWORD *)(v30 + 480);
          v66 = *(_QWORD *)(v52 + 32);
          if ( *v53 == v66 )
            goto LABEL_36;
          ++v53;
          goto LABEL_65;
        case 1LL:
          v52 = *(_QWORD *)(v30 + 480);
          v66 = *(_QWORD *)(v52 + 32);
          goto LABEL_70;
        default:
LABEL_53:
          if ( v68 == (const char *)++v29 )
            goto LABEL_54;
          break;
      }
    }
  }
LABEL_55:
  sub_2AC9810(v3, a2, *(_QWORD *)(*(_QWORD *)(v3 + 480) + 64LL));
  v64 = *(_QWORD *)(v3 + 240);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  return v64;
}
