// Function: sub_2ADCE40
// Address: 0x2adce40
//
__int64 __fastcall sub_2ADCE40(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // r14
  unsigned __int8 *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rcx
  __int64 v13; // rdx
  int v14; // esi
  __int64 v15; // r9
  __int64 v16; // r10
  unsigned __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned __int64 v19; // r9
  unsigned __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // edx
  unsigned int v26; // ecx
  __int64 v27; // r9
  __int64 v28; // rsi
  unsigned int v29; // edx
  __int64 v30; // r15
  __int64 v31; // rcx
  _QWORD *v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // r10
  _QWORD *v36; // rax
  const void *v37; // r10
  size_t v38; // rdx
  __int64 v39; // rax
  unsigned __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 *v42; // rdi
  unsigned __int8 *v43; // rax
  __int64 v44; // r9
  unsigned __int8 *v45; // r15
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // r13
  unsigned __int8 *v48; // rsi
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // r13
  __int64 v52; // rax
  unsigned __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v56; // rsi
  unsigned int v57; // r9d
  __int64 v58; // r11
  _QWORD *v59; // rdx
  _QWORD *v60; // r11
  __int64 v61; // rdx
  __int64 v62; // rbx
  __int64 v63; // r12
  unsigned __int64 v64; // r13
  unsigned __int64 v65; // rdi
  _QWORD *v66; // rax
  __int64 v68; // [rsp+10h] [rbp-80h]
  __int64 v69; // [rsp+18h] [rbp-78h]
  __int64 v70; // [rsp+18h] [rbp-78h]
  __int64 v71; // [rsp+20h] [rbp-70h]
  unsigned int v72; // [rsp+20h] [rbp-70h]
  __int64 v73; // [rsp+28h] [rbp-68h]
  __int64 v74; // [rsp+28h] [rbp-68h]
  __int64 v75; // [rsp+30h] [rbp-60h]
  unsigned __int64 v76; // [rsp+30h] [rbp-60h]
  __int64 v77; // [rsp+30h] [rbp-60h]
  __int64 v78; // [rsp+30h] [rbp-60h]
  __int64 v79; // [rsp+30h] [rbp-60h]
  __int64 v80; // [rsp+30h] [rbp-60h]
  __int64 v81; // [rsp+30h] [rbp-60h]
  __int64 v82; // [rsp+30h] [rbp-60h]
  __int64 v83; // [rsp+30h] [rbp-60h]
  __int64 v84; // [rsp+40h] [rbp-50h]
  __int64 v85; // [rsp+40h] [rbp-50h]
  __int64 v86; // [rsp+40h] [rbp-50h]
  __int64 v87; // [rsp+40h] [rbp-50h]
  unsigned int v88; // [rsp+40h] [rbp-50h]
  __int64 v89; // [rsp+40h] [rbp-50h]
  __int64 v91[7]; // [rsp+58h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 416);
  v3 = *(_QWORD *)(a1 + 240);
  if ( !*(_QWORD *)(v2 + 24) )
    return 0;
  v4 = a1;
  v5 = sub_AA54C0(*(_QWORD *)(a1 + 240));
  v6 = (unsigned __int8 *)sub_986580(v5);
  sub_B47210(v6, v3, *(_QWORD *)(v2 + 16));
  v7 = *(_QWORD *)(v2 + 32);
  if ( v5 )
  {
    v84 = *(_QWORD *)(v2 + 16);
    v8 = (unsigned int)(*(_DWORD *)(v5 + 44) + 1);
    if ( (unsigned int)(*(_DWORD *)(v5 + 44) + 1) < *(_DWORD *)(v7 + 32) )
      goto LABEL_4;
LABEL_74:
    *(_BYTE *)(v7 + 112) = 0;
    v79 = v7;
    v10 = sub_22077B0(0x50u);
    v12 = v79;
    v13 = v84;
    if ( !v10 )
    {
      v9 = 0;
      goto LABEL_8;
    }
    *(_QWORD *)v10 = v84;
    v9 = 0;
    v14 = 0;
    *(_QWORD *)(v10 + 8) = 0;
    goto LABEL_7;
  }
  v8 = 0;
  v84 = *(_QWORD *)(v2 + 16);
  if ( !*(_DWORD *)(v7 + 32) )
    goto LABEL_74;
LABEL_4:
  v75 = *(_QWORD *)(v2 + 32);
  v9 = *(_QWORD *)(*(_QWORD *)(v7 + 24) + 8 * v8);
  *(_BYTE *)(v7 + 112) = 0;
  v10 = sub_22077B0(0x50u);
  v12 = v75;
  v13 = v84;
  if ( v10 )
  {
    *(_QWORD *)v10 = v84;
    *(_QWORD *)(v10 + 8) = v9;
    if ( v9 )
      v14 = *(_DWORD *)(v9 + 16) + 1;
    else
      v14 = 0;
LABEL_7:
    *(_DWORD *)(v10 + 16) = v14;
    *(_QWORD *)(v10 + 24) = v10 + 40;
    *(_QWORD *)(v10 + 32) = 0x400000000LL;
    *(_QWORD *)(v10 + 72) = -1;
  }
LABEL_8:
  if ( v13 )
  {
    v15 = (unsigned int)(*(_DWORD *)(v13 + 44) + 1);
    v16 = 8 * v15;
  }
  else
  {
    v16 = 0;
    LODWORD(v15) = 0;
  }
  v17 = *(unsigned int *)(v12 + 32);
  if ( (unsigned int)v17 > (unsigned int)v15 )
    goto LABEL_11;
  v56 = *(_QWORD *)(v12 + 104);
  v57 = v15 + 1;
  if ( *(_DWORD *)(v56 + 88) >= v57 )
    v57 = *(_DWORD *)(v56 + 88);
  if ( v57 == v17 )
  {
LABEL_11:
    v18 = *(_QWORD *)(v12 + 24);
  }
  else
  {
    v58 = 8LL * v57;
    if ( v57 < v17 )
    {
      v18 = *(_QWORD *)(v12 + 24);
      v61 = v18 + 8 * v17;
      if ( v61 != v18 + v58 )
      {
        v89 = v3;
        v74 = v12;
        v72 = v57;
        v70 = v16;
        v68 = v10;
        v82 = v2;
        v62 = v18 + v58;
        v63 = v61;
        do
        {
          v64 = *(_QWORD *)(v63 - 8);
          v63 -= 8;
          if ( v64 )
          {
            v65 = *(_QWORD *)(v64 + 24);
            if ( v65 != v64 + 40 )
              _libc_free(v65);
            j_j___libc_free_0(v64);
          }
        }
        while ( v62 != v63 );
        v12 = v74;
        v2 = v82;
        v3 = v89;
        v57 = v72;
        v16 = v70;
        v10 = v68;
        v4 = a1;
        v18 = *(_QWORD *)(v74 + 24);
      }
    }
    else
    {
      if ( v57 > (unsigned __int64)*(unsigned int *)(v12 + 36) )
      {
        v69 = 8LL * v57;
        v71 = v10;
        v73 = v16;
        v88 = v57;
        v81 = v12;
        sub_B1B4E0(v12 + 24, v57);
        v12 = v81;
        v58 = v69;
        v10 = v71;
        v16 = v73;
        v17 = *(unsigned int *)(v81 + 32);
        v57 = v88;
      }
      v18 = *(_QWORD *)(v12 + 24);
      v59 = (_QWORD *)(v18 + 8 * v17);
      v60 = (_QWORD *)(v18 + v58);
      if ( v59 != v60 )
      {
        do
        {
          if ( v59 )
            *v59 = 0;
          ++v59;
        }
        while ( v60 != v59 );
        v18 = *(_QWORD *)(v12 + 24);
      }
    }
    *(_DWORD *)(v12 + 32) = v57;
  }
  v19 = *(_QWORD *)(v18 + v16);
  *(_QWORD *)(v18 + v16) = v10;
  if ( v19 )
  {
    v20 = *(_QWORD *)(v19 + 24);
    if ( v20 != v19 + 40 )
    {
      v85 = v10;
      v76 = v19;
      _libc_free(v20);
      v10 = v85;
      v19 = v76;
    }
    v77 = v10;
    j_j___libc_free_0(v19);
    v10 = v77;
  }
  if ( v9 )
  {
    v21 = *(unsigned int *)(v9 + 32);
    if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 36) )
    {
      v83 = v10;
      sub_C8D5F0(v9 + 24, (const void *)(v9 + 40), v21 + 1, 8u, v11, v21 + 1);
      v21 = *(unsigned int *)(v9 + 32);
      v10 = v83;
    }
    *(_QWORD *)(*(_QWORD *)(v9 + 24) + 8 * v21) = v10;
    ++*(_DWORD *)(v9 + 32);
  }
  v22 = *(_QWORD *)(v2 + 16);
  v23 = *(_QWORD *)(v2 + 32);
  if ( v22 )
  {
    v24 = (unsigned int)(*(_DWORD *)(v22 + 44) + 1);
    v25 = *(_DWORD *)(v22 + 44) + 1;
  }
  else
  {
    v24 = 0;
    v25 = 0;
  }
  v26 = *(_DWORD *)(v23 + 32);
  v27 = 0;
  if ( v25 < v26 )
    v27 = *(_QWORD *)(*(_QWORD *)(v23 + 24) + 8 * v24);
  if ( v3 )
  {
    v28 = (unsigned int)(*(_DWORD *)(v3 + 44) + 1);
    v29 = *(_DWORD *)(v3 + 44) + 1;
  }
  else
  {
    v28 = 0;
    v29 = 0;
  }
  if ( v26 <= v29 )
  {
    *(_BYTE *)(v23 + 112) = 0;
    BUG();
  }
  v30 = *(_QWORD *)(*(_QWORD *)(v23 + 24) + 8 * v28);
  *(_BYTE *)(v23 + 112) = 0;
  v31 = *(_QWORD *)(v30 + 8);
  if ( v27 != v31 )
  {
    v32 = *(_QWORD **)(v31 + 24);
    v33 = *(unsigned int *)(v31 + 32);
    v34 = (__int64)&v32[v33];
    v35 = (8 * v33) >> 3;
    if ( (8 * v33) >> 5 )
    {
      v36 = &v32[4 * ((8 * v33) >> 5)];
      while ( v30 != *v32 )
      {
        if ( v30 == v32[1] )
        {
          v37 = ++v32 + 1;
          goto LABEL_36;
        }
        if ( v30 == v32[2] )
        {
          v32 += 2;
          v37 = v32 + 1;
          goto LABEL_36;
        }
        if ( v30 == v32[3] )
        {
          v32 += 3;
          v37 = v32 + 1;
          goto LABEL_36;
        }
        v32 += 4;
        if ( v36 == v32 )
        {
          v35 = (v34 - (__int64)v32) >> 3;
          goto LABEL_80;
        }
      }
      goto LABEL_35;
    }
LABEL_80:
    switch ( v35 )
    {
      case 2LL:
        v66 = v32;
        break;
      case 3LL:
        v37 = v32 + 1;
        v66 = v32 + 1;
        if ( v30 == *v32 )
          goto LABEL_36;
        break;
      case 1LL:
        goto LABEL_87;
      default:
LABEL_83:
        v32 = (_QWORD *)v34;
        v37 = (const void *)(v34 + 8);
        goto LABEL_36;
    }
    v32 = v66 + 1;
    if ( v30 == *v66 )
    {
      v32 = v66;
      goto LABEL_35;
    }
LABEL_87:
    if ( v30 != *v32 )
      goto LABEL_83;
LABEL_35:
    v37 = v32 + 1;
LABEL_36:
    if ( v37 != (const void *)v34 )
    {
      v86 = *(_QWORD *)(v30 + 8);
      v38 = v34 - (_QWORD)v37;
      v34 = (__int64)v37;
      v78 = v27;
      memmove(v32, v37, v38);
      v31 = v86;
      v27 = v78;
      LODWORD(v33) = *(_DWORD *)(v86 + 32);
    }
    *(_DWORD *)(v31 + 32) = v33 - 1;
    *(_QWORD *)(v30 + 8) = v27;
    v39 = *(unsigned int *)(v27 + 32);
    v40 = *(unsigned int *)(v27 + 36);
    if ( v39 + 1 > v40 )
    {
      v34 = v27 + 40;
      v80 = v27;
      sub_C8D5F0(v27 + 24, (const void *)(v27 + 40), v39 + 1, 8u, v11, v27);
      v27 = v80;
      v39 = *(unsigned int *)(v80 + 32);
    }
    v41 = *(_QWORD *)(v27 + 24);
    *(_QWORD *)(v41 + 8 * v39) = v30;
    ++*(_DWORD *)(v27 + 32);
    if ( *(_DWORD *)(v30 + 16) != *(_DWORD *)(*(_QWORD *)(v30 + 8) + 16LL) + 1 )
      sub_2AA9740(v30, v34, v41, v40, v11, v27);
  }
  sub_AA4AC0(*(_QWORD *)(v2 + 16), v3 + 24);
  v42 = *(__int64 **)(v2 + 1808);
  if ( v42 )
    sub_D4F330(v42, *(_QWORD *)(v2 + 16), *(_QWORD *)(v2 + 40));
  v87 = *(_QWORD *)(v2 + 24);
  v43 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v45 = v43;
  if ( v43 )
    sub_B4C9A0((__int64)v43, a2, v3, v87, 3u, v44, 0, 0);
  if ( *(_BYTE *)(v2 + 1801) )
    sub_BC8EC0((__int64)v45, (unsigned int *)&unk_439F0C0, 2, 0);
  v46 = sub_986580(*(_QWORD *)(v2 + 16));
  sub_F34910(v46, v45);
  v47 = sub_986580(*(_QWORD *)(v2 + 16));
  v91[0] = *(_QWORD *)(sub_986580(v5) + 48);
  if ( v91[0] )
    sub_2AAAFA0(v91);
  if ( (__int64 *)(v47 + 48) != v91 )
  {
    sub_9C6650((_QWORD *)(v47 + 48));
    v48 = (unsigned __int8 *)v91[0];
    *(_QWORD *)(v47 + 48) = v91[0];
    if ( v48 )
    {
      sub_B976B0((__int64)v91, v48, v47 + 48);
      v91[0] = 0;
    }
  }
  sub_9C6650(v91);
  v51 = *(_QWORD *)(v2 + 16);
  *(_QWORD *)(v2 + 24) = 0;
  if ( !v51 )
    return 0;
  if ( *(_BYTE *)(*(_QWORD *)(v4 + 384) + 996LL) )
    sub_2AC29F0(*(__int64 **)(v4 + 64), v4);
  v52 = *(unsigned int *)(v4 + 272);
  v53 = *(unsigned int *)(v4 + 276);
  if ( v52 + 1 > v53 )
  {
    sub_C8D5F0(v4 + 264, (const void *)(v4 + 280), v52 + 1, 8u, v49, v50);
    v52 = *(unsigned int *)(v4 + 272);
  }
  v54 = *(_QWORD *)(v4 + 264);
  *(_QWORD *)(v54 + 8 * v52) = v51;
  ++*(_DWORD *)(v4 + 272);
  *(_BYTE *)(v4 + 392) = 1;
  sub_2AB95C0(v4, v51, v54, v53, v49, v50);
  return v51;
}
