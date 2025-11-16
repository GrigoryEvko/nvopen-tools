// Function: sub_2BD71E0
// Address: 0x2bd71e0
//
__int64 __fastcall sub_2BD71E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v5; // rdx
  unsigned int v7; // r12d
  int v9; // eax
  unsigned int v10; // ecx
  __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // rcx
  unsigned __int8 **v14; // r9
  unsigned __int8 **v15; // rdi
  unsigned __int8 **v16; // rbx
  __int64 v17; // rdi
  unsigned __int8 **v18; // r14
  unsigned int v19; // r13d
  int v20; // eax
  unsigned int v21; // esi
  __int64 v22; // rcx
  int v23; // r10d
  __int64 v24; // r14
  __int64 v25; // rbx
  __int64 v26; // r15
  int v27; // ecx
  int v28; // r8d
  unsigned int v29; // eax
  unsigned __int8 *v30; // rdi
  unsigned __int8 *v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // r15
  __int64 v34; // rbx
  __int64 v35; // r12
  int v36; // edx
  unsigned int v37; // eax
  __int64 v38; // rsi
  int v39; // eax
  __int64 v40; // r14
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 v48; // r14
  _BYTE *v49; // rdi
  int v50; // eax
  int v52; // r9d
  int v53; // edi
  __int64 v55; // [rsp+8h] [rbp-B8h]
  __int64 v56; // [rsp+18h] [rbp-A8h]
  __int64 v57; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v59; // [rsp+20h] [rbp-A0h]
  __int64 v60; // [rsp+28h] [rbp-98h]
  __int64 v61; // [rsp+30h] [rbp-90h] BYREF
  __int64 v62; // [rsp+38h] [rbp-88h] BYREF
  _QWORD v63[2]; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v64; // [rsp+50h] [rbp-70h] BYREF
  __int64 v65; // [rsp+58h] [rbp-68h]
  _BYTE v66[96]; // [rsp+60h] [rbp-60h] BYREF

  v55 = a2[1];
  v60 = *a2;
  if ( *a2 == v55 )
    return 0;
  v5 = *(_QWORD *)(a4 + 1984);
  v7 = 0;
  v9 = *(_DWORD *)(a4 + 2000);
  do
  {
    v12 = *(_QWORD *)(v60 - 8);
    if ( v9 )
    {
      v10 = (v9 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v11 = *(_QWORD *)(v5 + 8LL * v10);
      if ( v12 == v11 )
        goto LABEL_4;
      v52 = 1;
      while ( v11 != -4096 )
      {
        v10 = (v9 - 1) & (v52 + v10);
        v11 = *(_QWORD *)(v5 + 8LL * v10);
        if ( v12 == v11 )
          goto LABEL_4;
        ++v52;
      }
    }
    v13 = 4LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
    {
      v14 = *(unsigned __int8 ***)(v12 - 8);
      v15 = &v14[v13];
    }
    else
    {
      v15 = *(unsigned __int8 ***)(v60 - 8);
      v14 = (unsigned __int8 **)(v12 - v13 * 8);
    }
    if ( v14 != v15 )
    {
      v56 = *(_QWORD *)(v60 - 8);
      v16 = v15;
      v17 = a1;
      v18 = v14;
      v19 = v7;
      do
      {
        if ( **v18 > 0x1Cu )
        {
          v20 = sub_2BD7120(v17, 0, *v18, a3, a4, a5);
          v5 = *(_QWORD *)(a4 + 1984);
          v19 |= v20;
          v9 = *(_DWORD *)(a4 + 2000);
          if ( v9 )
          {
            v21 = (v9 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
            v22 = *(_QWORD *)(v5 + 8LL * v21);
            if ( v56 == v22 )
              break;
            v23 = 1;
            while ( v22 != -4096 )
            {
              v21 = (v9 - 1) & (v23 + v21);
              v22 = *(_QWORD *)(v5 + 8LL * v21);
              if ( v56 == v22 )
                goto LABEL_13;
              ++v23;
            }
          }
        }
        v18 += 4;
      }
      while ( v16 != v18 );
LABEL_13:
      v7 = v19;
      a1 = v17;
    }
LABEL_4:
    v60 -= 8;
  }
  while ( v55 != v60 );
  v24 = a4;
  v25 = *a2;
  v26 = a2[1];
  if ( v26 == *a2 )
    return v7;
  while ( 2 )
  {
    v31 = *(unsigned __int8 **)(v25 - 8);
    if ( v9 )
    {
      v27 = v9 - 1;
      v28 = 1;
      v29 = (v9 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v30 = *(unsigned __int8 **)(v5 + 8LL * v29);
      if ( v31 != v30 )
      {
        while ( v30 != (unsigned __int8 *)-4096LL )
        {
          v29 = v27 & (v28 + v29);
          v30 = *(unsigned __int8 **)(v5 + 8LL * v29);
          if ( v31 == v30 )
            goto LABEL_22;
          ++v28;
        }
        break;
      }
LABEL_22:
      v25 -= 8;
      if ( v26 == v25 )
        goto LABEL_26;
LABEL_23:
      v5 = *(_QWORD *)(v24 + 1984);
      v9 = *(_DWORD *)(v24 + 2000);
      continue;
    }
    break;
  }
  v7 |= sub_2BCF240(a1, v31, v24, a5);
  v25 -= 8;
  if ( v26 != v25 )
    goto LABEL_23;
LABEL_26:
  v61 = a1;
  v62 = a1;
  v32 = a2[1];
  v33 = *a2;
  v64 = v66;
  v65 = 0x600000000LL;
  if ( v32 == v33 )
    return v7;
  v59 = v7;
  v34 = v32;
  v35 = v24;
  v57 = a1;
  while ( 2 )
  {
    while ( 2 )
    {
      v39 = *(_DWORD *)(v35 + 2000);
      v40 = *(_QWORD *)(v33 - 8);
      v41 = *(_QWORD *)(v35 + 1984);
      if ( v39 )
      {
        v36 = v39 - 1;
        v37 = (v39 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v38 = *(_QWORD *)(v41 + 8LL * v37);
        if ( v40 != v38 )
        {
          v53 = 1;
          while ( v38 != -4096 )
          {
            v37 = v36 & (v53 + v37);
            v38 = *(_QWORD *)(v41 + 8LL * v37);
            if ( v40 == v38 )
              goto LABEL_29;
            ++v53;
          }
          break;
        }
LABEL_29:
        v33 -= 8;
        if ( v34 == v33 )
          goto LABEL_39;
        continue;
      }
      break;
    }
    v42 = sub_2B08520(*(char **)(v33 - 8));
    v43 = v42;
    if ( (_BYTE)qword_5010508 && *(_BYTE *)(v42 + 8) == 17 )
      v43 = **(_QWORD **)(v42 + 16);
    if ( !(unsigned __int8)sub_BCBCB0(v43) || (*(_BYTE *)(v43 + 8) & 0xFD) == 4 )
      goto LABEL_29;
    v46 = (unsigned int)v65;
    v47 = (unsigned int)v65 + 1LL;
    if ( v47 > HIDWORD(v65) )
    {
      sub_C8D5F0((__int64)&v64, v66, v47, 8u, v44, v45);
      v46 = (unsigned int)v65;
    }
    v33 -= 8;
    *(_QWORD *)&v64[8 * v46] = v40;
    LODWORD(v65) = v65 + 1;
    if ( v34 != v33 )
      continue;
    break;
  }
LABEL_39:
  v48 = v35;
  v7 = v59;
  v49 = v64;
  if ( (unsigned int)v65 > 1 )
  {
    v63[0] = v57;
    v63[1] = v48;
    v50 = sub_2BC5990(
            (__int64)&v64,
            (__int64 (__fastcall *)(__int64, __int64, __int64))sub_2B6F700,
            (__int64)&v61,
            (unsigned __int8 (__fastcall *)(__int64, unsigned __int8 *, unsigned __int8 *, __int64))sub_2B6F990,
            (__int64)&v62,
            v48,
            (__int64 (__fastcall *)(__int64, __int64 *, __int64, __int64))sub_2BCF930,
            (__int64)v63);
    v49 = v64;
    v7 = v50 | v59;
  }
  if ( v49 != v66 )
    _libc_free((unsigned __int64)v49);
  return v7;
}
