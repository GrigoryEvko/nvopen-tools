// Function: sub_303E210
// Address: 0x303e210
//
__int64 __fastcall sub_303E210(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r14
  __int128 v11; // rax
  int v12; // r9d
  __int64 v13; // rax
  int v14; // edx
  __int128 v15; // rax
  int v16; // r9d
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // r8
  int v20; // edx
  unsigned __int64 v21; // rdx
  int v22; // r15d
  _OWORD *v23; // rax
  _OWORD *i; // rdx
  unsigned __int64 v25; // r10
  const void *v26; // r9
  size_t v27; // r11
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  __int64 result; // rax
  _BYTE *v34; // rdi
  __int128 v35; // [rsp-20h] [rbp-160h]
  __int128 v36; // [rsp-20h] [rbp-160h]
  __int128 v37; // [rsp-10h] [rbp-150h]
  __int64 v38; // [rsp+0h] [rbp-140h]
  __int64 v39; // [rsp+8h] [rbp-138h]
  __int64 v40; // [rsp+10h] [rbp-130h]
  const void *v41; // [rsp+10h] [rbp-130h]
  __int64 v42; // [rsp+10h] [rbp-130h]
  int v43; // [rsp+18h] [rbp-128h]
  int v44; // [rsp+18h] [rbp-128h]
  unsigned __int64 v45; // [rsp+18h] [rbp-128h]
  int v46; // [rsp+20h] [rbp-120h]
  int v47; // [rsp+28h] [rbp-118h]
  __int64 v48; // [rsp+30h] [rbp-110h]
  __int64 v49; // [rsp+30h] [rbp-110h]
  __int64 v50; // [rsp+30h] [rbp-110h]
  __int64 v51; // [rsp+30h] [rbp-110h]
  __int64 v52; // [rsp+60h] [rbp-E0h] BYREF
  int v53; // [rsp+68h] [rbp-D8h]
  _BYTE *v54; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+78h] [rbp-C8h]
  _BYTE dest[48]; // [rsp+80h] [rbp-C0h] BYREF
  _OWORD *v57; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v58; // [rsp+B8h] [rbp-88h]
  _OWORD v59[8]; // [rsp+C0h] [rbp-80h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v52 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v52, v6, 1);
  v53 = *(_DWORD *)(a2 + 72);
  v7 = sub_33FB890(a4, 116, 0, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v9 = v8;
  v10 = v7;
  *(_QWORD *)&v11 = sub_3400D50(a4, 0, &v52, 0);
  *((_QWORD *)&v35 + 1) = v9;
  *(_QWORD *)&v35 = v10;
  v13 = sub_3406EB0(a4, 158, (unsigned int)&v52, 8, 0, v12, v35, v11);
  v47 = v14;
  v48 = v13;
  *(_QWORD *)&v15 = sub_3400D50(a4, 1, &v52, 0);
  *((_QWORD *)&v36 + 1) = v9;
  *(_QWORD *)&v36 = v10;
  v17 = sub_3406EB0(a4, 158, (unsigned int)&v52, 8, 0, v16, v36, v15);
  v57 = v59;
  v19 = v17;
  v46 = v20;
  v21 = (unsigned int)(*(_DWORD *)(a2 + 64) + 1);
  v58 = 0x500000000LL;
  v22 = v21;
  if ( v21 )
  {
    v23 = v59;
    if ( v21 > 5 )
    {
      v42 = v19;
      v45 = v21;
      sub_C8D5F0((__int64)&v57, v59, v21, 0x10u, v19, v18);
      v19 = v42;
      v23 = &v57[(unsigned int)v58];
      for ( i = &v57[v45]; i != v23; ++v23 )
      {
LABEL_6:
        if ( v23 )
        {
          *(_QWORD *)v23 = 0;
          *((_DWORD *)v23 + 2) = 0;
        }
      }
    }
    else
    {
      i = &v59[v21];
      if ( i != v59 )
        goto LABEL_6;
    }
    LODWORD(v58) = v22;
  }
  v25 = *(unsigned int *)(a2 + 68);
  v26 = *(const void **)(a2 + 48);
  v54 = dest;
  v55 = 0x300000000LL;
  v27 = 16 * v25;
  if ( v25 > 3 )
  {
    v38 = v19;
    v39 = 16 * v25;
    v41 = v26;
    v44 = v25;
    sub_C8D5F0((__int64)&v54, dest, v25, 0x10u, v19, (__int64)v26);
    LODWORD(v25) = v44;
    v26 = v41;
    v27 = v39;
    v19 = v38;
    v34 = &v54[16 * (unsigned int)v55];
  }
  else
  {
    if ( !v27 )
      goto LABEL_12;
    v34 = dest;
  }
  v40 = v19;
  v43 = v25;
  memcpy(v34, v26, v27);
  LODWORD(v27) = v55;
  v19 = v40;
  LODWORD(v25) = v43;
LABEL_12:
  v28 = *(_QWORD *)(a2 + 40);
  v29 = (unsigned __int64)v57;
  LODWORD(v55) = v27 + v25;
  *(_QWORD *)v57 = *(_QWORD *)v28;
  *(_DWORD *)(v29 + 8) = *(_DWORD *)(v28 + 8);
  v30 = *(_QWORD *)(a2 + 40);
  v31 = (unsigned __int64)v57;
  *((_QWORD *)v57 + 2) = *(_QWORD *)(v30 + 40);
  *(_DWORD *)(v31 + 24) = *(_DWORD *)(v30 + 48);
  *(_QWORD *)(v31 + 32) = v48;
  *(_DWORD *)(v31 + 40) = v47;
  *(_QWORD *)(v31 + 48) = v19;
  *(_DWORD *)(v31 + 56) = v46;
  if ( *(_DWORD *)(a2 + 64) == 4 )
  {
    v32 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(v31 + 64) = *(_QWORD *)(v32 + 120);
    *(_DWORD *)(v31 + 72) = *(_DWORD *)(v32 + 128);
  }
  *((_QWORD *)&v37 + 1) = (unsigned int)v58;
  *(_QWORD *)&v37 = v31;
  result = sub_3411BE0(a4, 49, (unsigned int)&v52, (_DWORD)v54, v55, (_DWORD)v26, v37);
  if ( v54 != dest )
  {
    v49 = result;
    _libc_free((unsigned __int64)v54);
    result = v49;
  }
  if ( v57 != v59 )
  {
    v50 = result;
    _libc_free((unsigned __int64)v57);
    result = v50;
  }
  if ( v52 )
  {
    v51 = result;
    sub_B91220((__int64)&v52, v52);
    return v51;
  }
  return result;
}
