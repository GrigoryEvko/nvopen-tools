// Function: sub_30FE720
// Address: 0x30fe720
//
__int64 __fastcall sub_30FE720(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  void **v7; // rax
  __int64 v10; // r13
  __int64 v11; // r14
  void **v12; // rdx
  __int64 v13; // rcx
  void **v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rdx
  float v18; // xmm0_4
  int v19; // eax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rcx
  int v23; // eax
  int v24; // esi
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rsi
  _QWORD *v29; // rdi
  _QWORD *v30; // rdx
  _QWORD *v31; // rax
  __int64 v32; // rcx
  unsigned int v33; // esi
  __int64 v34; // rdi
  unsigned int v35; // ecx
  __int64 *v36; // rdx
  __int64 result; // rax
  int v38; // r11d
  __int64 *v39; // r9
  int v40; // eax
  int v41; // edx
  void **v42; // rax
  __int64 *v43; // rax
  int v44; // eax
  int v45; // r8d
  int v46; // r8d
  int v47; // r8d
  __int64 v48; // r10
  __int64 v49; // rdi
  int v50; // esi
  __int64 *v51; // rcx
  int v52; // edi
  int v53; // edi
  __int64 v54; // r8
  __int64 *v55; // rsi
  unsigned int v56; // r15d
  __int64 v57; // rcx
  unsigned int v58; // r10d
  __int64 v59; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v60; // [rsp+18h] [rbp-88h]
  __int64 v61; // [rsp+20h] [rbp-80h]
  int v62; // [rsp+28h] [rbp-78h]
  char v63; // [rsp+2Ch] [rbp-74h]
  _QWORD v64[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v65; // [rsp+40h] [rbp-60h] BYREF
  void **v66; // [rsp+48h] [rbp-58h]
  __int64 v67; // [rsp+50h] [rbp-50h]
  int v68; // [rsp+58h] [rbp-48h]
  char v69; // [rsp+5Ch] [rbp-44h]
  void *v70; // [rsp+60h] [rbp-40h] BYREF

  v7 = (void **)v64;
  v66 = &v70;
  v10 = *(_QWORD *)(a2 + 16);
  v60 = v64;
  v67 = 2;
  v11 = *(_QWORD *)(a2 + 24);
  v68 = 0;
  v69 = 1;
  v61 = 0x100000002LL;
  v62 = 0;
  v63 = 1;
  v64[0] = &qword_4F82400;
  v59 = 1;
  if ( &qword_4F82400 == (__int64 *)&unk_502ED90 )
  {
    HIDWORD(v61) = 0;
    v59 = 2;
  }
  v70 = &unk_502ED90;
  HIDWORD(v67) = 1;
  v12 = (void **)&v64[HIDWORD(v61)];
  v65 = 1;
  if ( v12 == v64 )
  {
    v42 = v66;
    v13 = 1;
    v14 = v66 + 1;
LABEL_38:
    while ( *v42 != &unk_4F875F0 )
    {
      if ( ++v42 == v14 )
        goto LABEL_40;
    }
  }
  else
  {
    while ( *v7 != &unk_4F875F0 )
    {
      if ( v12 == ++v7 )
        goto LABEL_37;
    }
    --HIDWORD(v61);
    v13 = HIDWORD(v61);
    v14 = (void **)v64[HIDWORD(v61)];
    *v7 = v14;
    ++v59;
    if ( !v69 )
      goto LABEL_8;
LABEL_37:
    v42 = v66;
    v13 = HIDWORD(v67);
    v14 = &v66[HIDWORD(v67)];
    if ( v66 != v14 )
      goto LABEL_38;
LABEL_40:
    if ( (unsigned int)v67 <= (unsigned int)v13 )
    {
LABEL_8:
      sub_C8CC70((__int64)&v65, (__int64)&unk_4F875F0, (__int64)v14, v13, a5, a6);
    }
    else
    {
      v13 = (unsigned int)(v13 + 1);
      HIDWORD(v67) = v13;
      *v14 = &unk_4F875F0;
      ++v65;
    }
  }
  sub_BBE020(*(_QWORD *)(a1 + 16), v10, (__int64)&v59, v13);
  if ( !v69 )
    _libc_free((unsigned __int64)v66);
  if ( !v63 )
    _libc_free((unsigned __int64)v60);
  sub_30FC500(a2, *(_QWORD *)(a1 + 16));
  v15 = sub_30FCBF0(a1, v10);
  v16 = *(_QWORD *)(a2 + 72);
  v17 = v15[8];
  if ( !a3 )
    LODWORD(v17) = v16 + v17;
  v18 = (float)*(int *)(a1 + 248) * *(float *)&qword_5031728;
  v19 = v17 + *(_DWORD *)(a1 + 252) - *(_DWORD *)(a2 + 64) - v16;
  *(_DWORD *)(a1 + 252) = v19;
  if ( (float)v19 > v18 )
    *(_BYTE *)(a1 + 360) = 1;
  v20 = sub_30FCBF0(a1, v10)[3];
  if ( !a3 )
  {
    result = (__int64)sub_30FCBF0(a1, v11);
    v20 += *(_QWORD *)(result + 24);
    goto LABEL_43;
  }
  v21 = *(_QWORD *)(a1 + 168);
  --*(_QWORD *)(a1 + 176);
  v22 = *(_QWORD *)(v21 + 104);
  v23 = *(_DWORD *)(v21 + 120);
  if ( v23 )
  {
    v24 = v23 - 1;
    v25 = (v23 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v26 = (__int64 *)(v22 + 16LL * v25);
    v27 = *v26;
    if ( v11 == *v26 )
    {
LABEL_20:
      v28 = v26[1];
      if ( *(_BYTE *)(a1 + 284) )
        goto LABEL_21;
LABEL_45:
      v43 = sub_C8CA60(a1 + 256, v28);
      if ( v43 )
      {
        *v43 = -2;
        ++*(_DWORD *)(a1 + 280);
        ++*(_QWORD *)(a1 + 256);
      }
      goto LABEL_26;
    }
    v44 = 1;
    while ( v27 != -4096 )
    {
      v45 = v44 + 1;
      v25 = v24 & (v44 + v25);
      v26 = (__int64 *)(v22 + 16LL * v25);
      v27 = *v26;
      if ( v11 == *v26 )
        goto LABEL_20;
      v44 = v45;
    }
  }
  v28 = 0;
  if ( !*(_BYTE *)(a1 + 284) )
    goto LABEL_45;
LABEL_21:
  v29 = *(_QWORD **)(a1 + 264);
  v30 = &v29[*(unsigned int *)(a1 + 276)];
  v31 = v29;
  if ( v29 != v30 )
  {
    while ( *v31 != v28 )
    {
      if ( v30 == ++v31 )
        goto LABEL_26;
    }
    v32 = (unsigned int)(*(_DWORD *)(a1 + 276) - 1);
    *(_DWORD *)(a1 + 276) = v32;
    *v31 = v29[v32];
    ++*(_QWORD *)(a1 + 256);
  }
LABEL_26:
  v33 = *(_DWORD *)(a1 + 352);
  if ( !v33 )
  {
    ++*(_QWORD *)(a1 + 328);
    goto LABEL_53;
  }
  v34 = *(_QWORD *)(a1 + 336);
  v35 = (v33 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v36 = (__int64 *)(v34 + 8LL * v35);
  result = *v36;
  if ( v11 == *v36 )
    goto LABEL_43;
  v38 = 1;
  v39 = 0;
  while ( result != -4096 )
  {
    if ( v39 || result != -8192 )
      v36 = v39;
    v35 = (v33 - 1) & (v38 + v35);
    result = *(_QWORD *)(v34 + 8LL * v35);
    if ( v11 == result )
      goto LABEL_43;
    ++v38;
    v39 = v36;
    v36 = (__int64 *)(v34 + 8LL * v35);
  }
  v40 = *(_DWORD *)(a1 + 344);
  if ( !v39 )
    v39 = v36;
  ++*(_QWORD *)(a1 + 328);
  v41 = v40 + 1;
  if ( 4 * (v40 + 1) >= 3 * v33 )
  {
LABEL_53:
    sub_A35F10(a1 + 328, 2 * v33);
    v46 = *(_DWORD *)(a1 + 352);
    if ( v46 )
    {
      v47 = v46 - 1;
      v48 = *(_QWORD *)(a1 + 336);
      result = v47 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v39 = (__int64 *)(v48 + 8 * result);
      v41 = *(_DWORD *)(a1 + 344) + 1;
      v49 = *v39;
      if ( v11 != *v39 )
      {
        v50 = 1;
        v51 = 0;
        while ( v49 != -4096 )
        {
          if ( v49 == -8192 && !v51 )
            v51 = v39;
          result = v47 & (unsigned int)(v50 + result);
          v39 = (__int64 *)(v48 + 8LL * (unsigned int)result);
          v49 = *v39;
          if ( v11 == *v39 )
            goto LABEL_34;
          ++v50;
        }
        if ( v51 )
          v39 = v51;
      }
      goto LABEL_34;
    }
    goto LABEL_81;
  }
  result = v33 - *(_DWORD *)(a1 + 348) - v41;
  if ( (unsigned int)result <= v33 >> 3 )
  {
    sub_A35F10(a1 + 328, v33);
    v52 = *(_DWORD *)(a1 + 352);
    if ( v52 )
    {
      v53 = v52 - 1;
      v54 = *(_QWORD *)(a1 + 336);
      v55 = 0;
      v56 = v53 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v39 = (__int64 *)(v54 + 8LL * v56);
      v57 = *v39;
      v41 = *(_DWORD *)(a1 + 344) + 1;
      result = 1;
      if ( v11 != *v39 )
      {
        while ( v57 != -4096 )
        {
          if ( v57 == -8192 && !v55 )
            v55 = v39;
          v58 = result + 1;
          result = v53 & (v56 + (unsigned int)result);
          v39 = (__int64 *)(v54 + 8 * result);
          v56 = result;
          v57 = *v39;
          if ( v11 == *v39 )
            goto LABEL_34;
          result = v58;
        }
        if ( v55 )
          v39 = v55;
      }
      goto LABEL_34;
    }
LABEL_81:
    ++*(_DWORD *)(a1 + 344);
    BUG();
  }
LABEL_34:
  *(_DWORD *)(a1 + 344) = v41;
  if ( *v39 != -4096 )
    --*(_DWORD *)(a1 + 348);
  *v39 = v11;
LABEL_43:
  *(_QWORD *)(a1 + 184) += v20 - *(_QWORD *)(a2 + 80);
  return result;
}
