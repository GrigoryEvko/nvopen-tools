// Function: sub_2F385E0
// Address: 0x2f385e0
//
__int64 __fastcall sub_2F385E0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r9
  __int64 v8; // rdx
  int v9; // ebx
  unsigned int i; // eax
  __int64 v11; // r10
  unsigned int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // r13d
  unsigned int j; // eax
  __int64 v18; // r11
  unsigned int v19; // eax
  __int64 v20; // rax
  int v21; // r13d
  unsigned int k; // eax
  __int64 v23; // r11
  unsigned int v24; // eax
  __int64 v25; // rax
  int v26; // ebx
  unsigned int m; // eax
  __int64 v28; // r10
  unsigned int v29; // eax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  _BYTE v51[8]; // [rsp+0h] [rbp-150h] BYREF
  unsigned __int64 v52; // [rsp+8h] [rbp-148h]
  char v53; // [rsp+1Ch] [rbp-134h]
  _BYTE v54[16]; // [rsp+20h] [rbp-130h] BYREF
  _BYTE v55[8]; // [rsp+30h] [rbp-120h] BYREF
  unsigned __int64 v56; // [rsp+38h] [rbp-118h]
  char v57; // [rsp+4Ch] [rbp-104h]
  _BYTE v58[16]; // [rsp+50h] [rbp-100h] BYREF
  __int64 v59; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v60; // [rsp+68h] [rbp-E8h]
  __int64 v61; // [rsp+70h] [rbp-E0h]
  __int64 v62; // [rsp+78h] [rbp-D8h]
  __int64 v63; // [rsp+80h] [rbp-D0h]
  __int64 v64; // [rsp+88h] [rbp-C8h]
  __int64 v65; // [rsp+90h] [rbp-C0h]
  __int64 v66; // [rsp+98h] [rbp-B8h]
  unsigned int v67; // [rsp+A0h] [rbp-B0h]
  __int64 v68; // [rsp+A8h] [rbp-A8h]
  char *v69; // [rsp+B0h] [rbp-A0h]
  __int64 v70; // [rsp+B8h] [rbp-98h]
  int v71; // [rsp+C0h] [rbp-90h]
  char v72; // [rsp+C4h] [rbp-8Ch]
  char v73; // [rsp+C8h] [rbp-88h] BYREF
  __int64 v74; // [rsp+E8h] [rbp-68h]
  __int64 v75; // [rsp+F0h] [rbp-60h]
  __int64 v76; // [rsp+F8h] [rbp-58h]
  unsigned int v77; // [rsp+100h] [rbp-50h]
  __int64 v78; // [rsp+108h] [rbp-48h]
  __int64 v79; // [rsp+110h] [rbp-40h]

  v7 = *(unsigned int *)(a4 + 88);
  v8 = *(_QWORD *)(a4 + 72);
  v59 = 0;
  if ( (_DWORD)v7 )
  {
    v9 = 1;
    a5 = (unsigned int)(v7 - 1);
    for ( i = a5
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_501EB18 >> 9) ^ ((unsigned int)&unk_501EB18 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = a5 & v12 )
    {
      v11 = v8 + 24LL * i;
      if ( *(_UNKNOWN **)v11 == &unk_501EB18 && a3 == *(_QWORD **)(v11 + 8) )
        break;
      if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
        goto LABEL_7;
      v12 = v9 + i;
      ++v9;
    }
    v13 = v8 + 24LL * (unsigned int)v7;
    if ( v13 == v11 )
    {
      v60 = 0;
    }
    else
    {
      v15 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL);
      if ( v15 )
        v15 += 8;
      v60 = v15;
    }
  }
  else
  {
LABEL_7:
    v60 = 0;
    v13 = v8 + 24LL * (unsigned int)v7;
    if ( !(_DWORD)v7 )
      goto LABEL_8;
    a5 = (unsigned int)(v7 - 1);
  }
  v16 = 1;
  for ( j = a5
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_501EAD0 >> 9) ^ ((unsigned int)&unk_501EAD0 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = a5 & v19 )
  {
    v18 = v8 + 24LL * j;
    if ( *(_UNKNOWN **)v18 == &unk_501EAD0 && a3 == *(_QWORD **)(v18 + 8) )
      break;
    if ( *(_QWORD *)v18 == -4096 && *(_QWORD *)(v18 + 8) == -4096 )
      goto LABEL_8;
    v19 = v16 + j;
    ++v16;
  }
  if ( v18 != v13 )
  {
    v20 = *(_QWORD *)(*(_QWORD *)(v18 + 16) + 24LL);
    if ( v20 )
      v20 += 8;
    v61 = v20;
    goto LABEL_25;
  }
LABEL_8:
  v61 = 0;
  if ( !(_DWORD)v7 )
    goto LABEL_9;
  a5 = (unsigned int)(v7 - 1);
LABEL_25:
  v21 = 1;
  for ( k = a5
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&qword_50208B0 >> 9) ^ ((unsigned int)&qword_50208B0 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; k = a5 & v24 )
  {
    v23 = v8 + 24LL * k;
    if ( *(__int64 **)v23 == &qword_50208B0 && a3 == *(_QWORD **)(v23 + 8) )
      break;
    if ( *(_QWORD *)v23 == -4096 && *(_QWORD *)(v23 + 8) == -4096 )
      goto LABEL_9;
    v24 = v21 + k;
    ++v21;
  }
  if ( v23 != v13 )
  {
    v25 = *(_QWORD *)(*(_QWORD *)(v23 + 16) + 24LL);
    if ( v25 )
      v25 += 8;
    v62 = v25;
    goto LABEL_34;
  }
LABEL_9:
  v62 = 0;
  if ( !(_DWORD)v7 )
  {
LABEL_10:
    v14 = 0;
    goto LABEL_42;
  }
  a5 = (unsigned int)(v7 - 1);
LABEL_34:
  v7 = (__int64)qword_501FE48;
  v26 = 1;
  for ( m = a5
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)qword_501FE48 >> 9) ^ ((unsigned int)qword_501FE48 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; m = a5 & v29 )
  {
    v28 = v8 + 24LL * m;
    if ( *(__int64 **)v28 == qword_501FE48 && a3 == *(_QWORD **)(v28 + 8) )
      break;
    if ( *(_QWORD *)v28 == -4096 && *(_QWORD *)(v28 + 8) == -4096 )
      goto LABEL_10;
    v29 = v26 + m;
    ++v26;
  }
  if ( v28 == v13 )
    goto LABEL_10;
  v14 = *(_QWORD *)(*(_QWORD *)(v28 + 16) + 24LL);
  v8 = v14 + 8;
  if ( v14 )
    v14 += 8;
LABEL_42:
  v63 = v14;
  v79 = a4;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = &v73;
  v70 = 4;
  v71 = 0;
  v72 = 1;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  if ( (unsigned __int8)sub_2F36310((__int64)&v59, a3, v8, a4, a5, v7) )
  {
    sub_2EAFFB0((__int64)v51);
    sub_2F31C10((__int64)v51, (__int64)&unk_501EAD0, v31, v32, v33, v34);
    sub_2F31C10((__int64)v51, (__int64)&unk_501EB18, v35, v36, v37, v38);
    sub_2F31C10((__int64)v51, (__int64)&unk_5025C20, v39, v40, v41, v42);
    sub_2F31C10((__int64)v51, (__int64)qword_501FE48, v43, v44, v45, v46);
    sub_2F31C10((__int64)v51, (__int64)&qword_50208B0, v47, v48, v49, v50);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v54, (__int64)v51);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v58, (__int64)v55);
    if ( !v57 )
      _libc_free(v56);
    if ( !v53 )
      _libc_free(v52);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  sub_C7D6A0(v75, 16LL * v77, 8);
  if ( !v72 )
    _libc_free((unsigned __int64)v69);
  sub_C7D6A0(v65, 12LL * v67, 4);
  return a1;
}
