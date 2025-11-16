// Function: sub_279BB20
// Address: 0x279bb20
//
__int64 __fastcall sub_279BB20(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  int v9; // ecx
  __int64 v10; // rdi
  int v11; // ecx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r8
  __int64 v15; // r13
  unsigned int v16; // r15d
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int8 v20; // r15
  __int64 *v21; // r8
  __int64 v22; // rax
  __int64 *v23; // r15
  __int64 v24; // rbx
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rax
  int v28; // esi
  __int64 v29; // rdi
  int v30; // esi
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // r10
  unsigned __int64 v34; // rdi
  int v35; // eax
  __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  int v45; // eax
  int v46; // r9d
  int v47; // eax
  int v48; // ecx
  unsigned __int8 v49; // [rsp+17h] [rbp-99h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  __int64 v51; // [rsp+20h] [rbp-90h]
  __int64 *v52; // [rsp+30h] [rbp-80h]
  __int64 v54; // [rsp+40h] [rbp-70h] BYREF
  __int64 v55; // [rsp+48h] [rbp-68h] BYREF
  __int64 v56; // [rsp+50h] [rbp-60h] BYREF
  __int64 v57; // [rsp+58h] [rbp-58h]
  __int64 v58; // [rsp+60h] [rbp-50h]
  unsigned int v59; // [rsp+68h] [rbp-48h]
  _BYTE *v60; // [rsp+70h] [rbp-40h]
  __int64 v61; // [rsp+78h] [rbp-38h]
  _BYTE v62[48]; // [rsp+80h] [rbp-30h] BYREF

  v7 = a1[14];
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_DWORD *)(v7 + 24);
  v10 = *(_QWORD *)(v7 + 8);
  if ( !v9 )
    return 0;
  v11 = v9 - 1;
  v12 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v13 = (__int64 *)(v10 + 16LL * v12);
  v14 = *v13;
  if ( v8 != *v13 )
  {
    v45 = 1;
    while ( v14 != -4096 )
    {
      v46 = v45 + 1;
      v12 = v11 & (v45 + v12);
      v13 = (__int64 *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
        goto LABEL_3;
      v45 = v46;
    }
    return 0;
  }
LABEL_3:
  v15 = v13[1];
  if ( !v15 )
    return 0;
  if ( **(_QWORD **)(v15 + 32) != v8 )
    return 0;
  v54 = sub_D4B130(v15);
  v18 = sub_D47930(v15);
  if ( !v54 )
    return 0;
  v51 = v18;
  if ( !v18 )
    return 0;
  v50 = *(_QWORD *)(a2 - 32);
  v20 = sub_D48480(v15, v50, v19, v50);
  if ( !v20 )
    return 0;
  if ( (unsigned __int8)sub_30ED170(a1[13], a2) )
    return 0;
  v21 = *(__int64 **)a4;
  v22 = *(unsigned int *)(a4 + 8);
  v55 = 0;
  v52 = &v21[v22];
  if ( v52 == v21 )
    return 0;
  v49 = v20;
  v23 = v21;
  do
  {
    v24 = *v23;
    if ( *(_BYTE *)(v15 + 84) )
    {
      v25 = *(_QWORD **)(v15 + 64);
      v26 = &v25[*(unsigned int *)(v15 + 76)];
      if ( v25 == v26 )
        goto LABEL_29;
      while ( v24 != *v25 )
      {
        if ( v26 == ++v25 )
          goto LABEL_29;
      }
    }
    else if ( !sub_C8CA60(v15 + 56, *v23) )
    {
      goto LABEL_29;
    }
    if ( v55 )
      return 0;
    v27 = a1[14];
    v28 = *(_DWORD *)(v27 + 24);
    v29 = *(_QWORD *)(v27 + 8);
    if ( !v28 )
      return 0;
    v30 = v28 - 1;
    v31 = v30 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v32 = (__int64 *)(v29 + 16LL * v31);
    v33 = *v32;
    if ( v24 != *v32 )
    {
      v47 = 1;
      while ( v33 != -4096 )
      {
        v48 = v47 + 1;
        v31 = v30 & (v47 + v31);
        v32 = (__int64 *)(v29 + 16LL * v31);
        v33 = *v32;
        if ( v24 == *v32 )
          goto LABEL_21;
        v47 = v48;
      }
      return 0;
    }
LABEL_21:
    if ( v15 != v32[1] || (unsigned __int8)sub_B19720(a1[3], v24, v51) )
      return 0;
    v34 = *(_QWORD *)(v24 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v34 == v24 + 48 )
    {
      v36 = 0;
    }
    else
    {
      if ( !v34 )
        BUG();
      v35 = *(unsigned __int8 *)(v34 - 24);
      v36 = v34 - 24;
      if ( (unsigned int)(v35 - 30) >= 0xB )
        v36 = 0;
    }
    if ( (unsigned __int8)sub_B46490(v36) )
      return 0;
    v55 = v24;
LABEL_29:
    ++v23;
  }
  while ( v52 != v23 );
  v16 = v49;
  if ( !v55 || (unsigned __int8)sub_BD4ED0(v50) )
    return 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = v62;
  v61 = 0;
  *(_QWORD *)sub_2799C30((__int64)&v56, &v55, v37, v38, v39, v40) = v50;
  *(_QWORD *)sub_2799C30((__int64)&v56, &v54, v41, v42, v43, v44) = v50;
  sub_2791B90((__int64)a1, a2, a3, (__int64)&v56, 0);
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  sub_C7D6A0(v57, 16LL * v59, 8);
  return v16;
}
