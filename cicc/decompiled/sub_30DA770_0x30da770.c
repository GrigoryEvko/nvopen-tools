// Function: sub_30DA770
// Address: 0x30da770
//
__int64 __fastcall sub_30DA770(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r14
  bool v9; // al
  int v10; // r9d
  __int64 v11; // r9
  unsigned __int8 **v12; // rbx
  _QWORD *v13; // rax
  _BYTE *v14; // rcx
  int v15; // edx
  __int64 v16; // r8
  unsigned int v17; // edi
  _QWORD *v18; // rsi
  _BYTE *v19; // r15
  int v20; // eax
  __int64 v21; // rax
  unsigned __int64 v22; // r9
  __int64 v23; // rcx
  unsigned __int8 *v24; // r15
  int v25; // edx
  unsigned int v26; // esi
  unsigned __int8 **v27; // rax
  unsigned __int8 *v28; // r11
  unsigned __int8 **v29; // rdx
  unsigned __int8 *v30; // rsi
  unsigned int v31; // r12d
  __int64 v32; // rax
  int v33; // edx
  int v34; // ecx
  __int64 v35; // rdi
  __int64 v36; // rsi
  int v37; // ecx
  unsigned int v38; // edx
  __int64 *v39; // rax
  __int64 v40; // r8
  __int64 v41; // rcx
  __int64 *v42; // rax
  bool v43; // cc
  int v45; // eax
  int v46; // r10d
  unsigned __int8 *v47; // rax
  int v48; // esi
  int v49; // r11d
  int v50; // eax
  int v51; // r10d
  unsigned __int8 *v52; // [rsp+0h] [rbp-80h]
  __int64 v53; // [rsp+18h] [rbp-68h] BYREF
  unsigned __int8 **v54; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int64 v55; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v56[10]; // [rsp+30h] [rbp-50h] BYREF

  v8 = sub_30D1740(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( !byte_502FB48 && (unsigned __int8)sub_30D92D0(a1, a2, v4, v5, v6, v7) )
    return 1;
  v9 = sub_B4DE30(a2);
  v10 = *(_DWORD *)(a2 + 4);
  if ( !v9 )
    goto LABEL_4;
  v34 = *(_DWORD *)(a1 + 256);
  v35 = *(_QWORD *)(a1 + 240);
  v36 = *(_QWORD *)(a2 - 32LL * (v10 & 0x7FFFFFF));
  if ( !v34 )
    goto LABEL_4;
  v37 = v34 - 1;
  v38 = v37 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
  v39 = (__int64 *)(v35 + 32LL * v38);
  v40 = *v39;
  if ( v36 != *v39 )
  {
    v50 = 1;
    while ( v40 != -4096 )
    {
      v51 = v50 + 1;
      v38 = v37 & (v50 + v38);
      v39 = (__int64 *)(v35 + 32LL * v38);
      v40 = *v39;
      if ( v36 == *v39 )
        goto LABEL_32;
      v50 = v51;
    }
    goto LABEL_4;
  }
LABEL_32:
  v41 = v39[1];
  v54 = (unsigned __int8 **)v41;
  LODWORD(v56[0]) = *((_DWORD *)v39 + 6);
  if ( LODWORD(v56[0]) > 0x40 )
  {
    sub_C43780((__int64)&v55, (const void **)v39 + 2);
    if ( !v54 )
    {
LABEL_49:
      if ( LODWORD(v56[0]) > 0x40 && v55 )
        j_j___libc_free_0_0(v55);
      v10 = *(_DWORD *)(a2 + 4);
      goto LABEL_4;
    }
LABEL_34:
    if ( (unsigned __int8)sub_30D43C0(a1, a2, (__int64)&v55) )
    {
      v53 = a2;
      v42 = sub_30DA4E0(a1 + 232, &v53);
      v43 = *((_DWORD *)v42 + 4) <= 0x40u;
      *v42 = (__int64)v54;
      if ( v43 && LODWORD(v56[0]) <= 0x40 )
      {
        v42[1] = v55;
        *((_DWORD *)v42 + 4) = v56[0];
      }
      else
      {
        sub_C43990((__int64)(v42 + 1), (__int64)&v55);
      }
      if ( LODWORD(v56[0]) > 0x40 && v55 )
        j_j___libc_free_0_0(v55);
      goto LABEL_39;
    }
    goto LABEL_49;
  }
  v55 = v39[2];
  if ( v41 )
    goto LABEL_34;
LABEL_4:
  v11 = v10 & 0x7FFFFFF;
  v12 = (unsigned __int8 **)(a2 + 32 * (1 - v11));
  if ( (unsigned __int8 **)a2 == v12 )
  {
LABEL_39:
    if ( v8 )
    {
      v54 = (unsigned __int8 **)a2;
      *sub_30DA630(a1 + 168, (__int64 *)&v54) = v8;
    }
    return 1;
  }
  v13 = (_QWORD *)(a2 + 32 * (1 - v11));
  while ( 1 )
  {
    v14 = (_BYTE *)*v13;
    if ( *(_BYTE *)*v13 <= 0x15u )
      goto LABEL_6;
    v15 = *(_DWORD *)(a1 + 160);
    v16 = *(_QWORD *)(a1 + 144);
    if ( !v15 )
      goto LABEL_11;
    v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v18 = (_QWORD *)(v16 + 16LL * v17);
    v19 = (_BYTE *)*v18;
    if ( v14 != (_BYTE *)*v18 )
      break;
LABEL_10:
    if ( !v18[1] )
      goto LABEL_11;
LABEL_6:
    v13 += 4;
    if ( (_QWORD *)a2 == v13 )
      goto LABEL_39;
  }
  v48 = 1;
  while ( v19 != (_BYTE *)-4096LL )
  {
    v49 = v48 + 1;
    v17 = (v15 - 1) & (v48 + v17);
    v18 = (_QWORD *)(v16 + 16LL * v17);
    v19 = (_BYTE *)*v18;
    if ( v14 == (_BYTE *)*v18 )
      goto LABEL_10;
    v48 = v49;
  }
LABEL_11:
  if ( !v8 )
  {
    v54 = (unsigned __int8 **)v56;
    v56[0] = *(_QWORD *)(a2 - 32 * v11);
    v55 = 0x400000001LL;
LABEL_14:
    v22 = 4;
    v23 = 1;
    while ( 1 )
    {
      v24 = *v12;
      if ( !v15 )
        goto LABEL_22;
      v25 = v15 - 1;
      v26 = v25 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v27 = (unsigned __int8 **)(v16 + 16LL * v26);
      v28 = *v27;
      if ( *v27 != v24 )
        break;
LABEL_21:
      v47 = v27[1];
      if ( !v47 )
        goto LABEL_22;
      if ( v23 + 1 > v22 )
      {
        v52 = v47;
        sub_C8D5F0((__int64)&v54, v56, v23 + 1, 8u, v16, v22);
        v23 = (unsigned int)v55;
        v47 = v52;
      }
      v12 += 4;
      v54[v23] = v47;
      v23 = (unsigned int)(v55 + 1);
      LODWORD(v55) = v55 + 1;
      if ( (unsigned __int8 **)a2 == v12 )
      {
LABEL_25:
        v29 = v54;
        goto LABEL_26;
      }
LABEL_18:
      v22 = HIDWORD(v55);
      v16 = *(_QWORD *)(a1 + 144);
      v15 = *(_DWORD *)(a1 + 160);
    }
    v45 = 1;
    while ( v28 != (unsigned __int8 *)-4096LL )
    {
      v46 = v45 + 1;
      v26 = v25 & (v45 + v26);
      v27 = (unsigned __int8 **)(v16 + 16LL * v26);
      v28 = *v27;
      if ( v24 == *v27 )
        goto LABEL_21;
      v45 = v46;
    }
LABEL_22:
    if ( v23 + 1 > v22 )
    {
      sub_C8D5F0((__int64)&v54, v56, v23 + 1, 8u, v16, v22);
      v23 = (unsigned int)v55;
    }
    v12 += 4;
    v54[v23] = v24;
    v23 = (unsigned int)(v55 + 1);
    LODWORD(v55) = v55 + 1;
    if ( (unsigned __int8 **)a2 == v12 )
      goto LABEL_25;
    goto LABEL_18;
  }
  sub_30D1890(a1, v8);
  v20 = *(_DWORD *)(a2 + 4);
  v54 = (unsigned __int8 **)v56;
  v21 = v20 & 0x7FFFFFF;
  v12 = (unsigned __int8 **)(a2 + 32 * (1 - v21));
  v56[0] = *(_QWORD *)(a2 - 32 * v21);
  v55 = 0x400000001LL;
  if ( (unsigned __int8 **)a2 != v12 )
  {
    v16 = *(_QWORD *)(a1 + 144);
    v15 = *(_DWORD *)(a1 + 160);
    goto LABEL_14;
  }
  v23 = 1;
  v29 = (unsigned __int8 **)v56;
LABEL_26:
  v30 = (unsigned __int8 *)a2;
  v31 = 0;
  v32 = sub_DFCEF0(*(__int64 ***)(a1 + 8), v30, v29, v23, 3);
  if ( !v33 )
    LOBYTE(v31) = v32 == 0;
  if ( v54 != v56 )
    _libc_free((unsigned __int64)v54);
  return v31;
}
