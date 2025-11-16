// Function: sub_3097E10
// Address: 0x3097e10
//
void __fastcall sub_3097E10(
        __int64 a1,
        _DWORD *a2,
        __int64 *a3,
        __int64 (__fastcall ***a4)(_QWORD, __int64),
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v7; // rbx
  __int64 v8; // r13
  _BYTE *v9; // rdx
  int v10; // ecx
  _BYTE *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  int v18; // ebx
  _BYTE *v19; // rcx
  __int64 v20; // r13
  _BYTE *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rax
  int v31; // ebx
  _BYTE *v32; // rcx
  __int64 v33; // r13
  _BYTE *v34; // rsi
  __int64 v35; // rdx
  _QWORD *v36; // r13
  _QWORD *v37; // rbx
  unsigned __int64 v38; // r14
  unsigned __int64 v39; // rdi
  _QWORD *v40; // rax
  _QWORD *v41; // r14
  unsigned __int64 v42; // r13
  unsigned __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  unsigned __int64 v48; // rdi
  __int64 v49; // rax
  unsigned __int64 v50; // rdi
  __int64 v51; // rax
  unsigned int v55; // [rsp+48h] [rbp-5B8h] BYREF
  int v56; // [rsp+4Ch] [rbp-5B4h] BYREF
  void *v57; // [rsp+50h] [rbp-5B0h] BYREF
  _BYTE *v58; // [rsp+58h] [rbp-5A8h]
  __int64 v59; // [rsp+60h] [rbp-5A0h]
  _BYTE v60[128]; // [rsp+68h] [rbp-598h] BYREF
  __int64 v61; // [rsp+E8h] [rbp-518h]
  __int64 v62; // [rsp+F0h] [rbp-510h]
  __int64 v63; // [rsp+F8h] [rbp-508h]
  unsigned int v64; // [rsp+100h] [rbp-500h]
  _BYTE *v65; // [rsp+108h] [rbp-4F8h]
  __int64 v66; // [rsp+110h] [rbp-4F0h]
  _BYTE v67[128]; // [rsp+118h] [rbp-4E8h] BYREF
  _QWORD *v68; // [rsp+198h] [rbp-468h] BYREF
  _QWORD **v69; // [rsp+1A0h] [rbp-460h]
  __int64 v70; // [rsp+1A8h] [rbp-458h]
  int v71; // [rsp+1B0h] [rbp-450h]
  _BYTE *v72; // [rsp+1C0h] [rbp-440h] BYREF
  __int64 v73; // [rsp+1C8h] [rbp-438h]
  _BYTE v74[1072]; // [rsp+1D0h] [rbp-430h] BYREF

  v7 = (int)*a2;
  v72 = v74;
  v8 = *a3;
  v73 = 0x8000000000LL;
  if ( 8 * v7 > 0x400 )
  {
    sub_C8D5F0((__int64)&v72, v74, v7, 8u, a5, a6);
    v9 = v72;
    v10 = v73;
    v11 = &v72[8 * (unsigned int)v73];
  }
  else
  {
    v9 = v74;
    v10 = 0;
    v11 = v74;
  }
  if ( (__int64)(8 * v7) > 0 )
  {
    v12 = 0;
    do
    {
      *(_QWORD *)&v11[8 * v12] = *(_QWORD *)(v8 + 8 * v12);
      ++v12;
    }
    while ( (__int64)(v7 - v12) > 0 );
    v9 = v72;
    v10 = v73;
  }
  LODWORD(v73) = v10 + v7;
  sub_22F7AF0((__int64)&v57, a1, v9, (unsigned int)(v10 + v7), &v55, &v56, -1);
  sub_22F3580(a1 + 184);
  v15 = (unsigned __int64)v58;
  if ( v58 == v60 )
  {
    v16 = (unsigned int)v59;
    v17 = *(unsigned int *)(a1 + 200);
    v18 = v59;
    if ( (unsigned int)v59 <= v17 )
    {
      if ( (_DWORD)v59 )
        memmove(*(void **)(a1 + 192), v60, 8LL * (unsigned int)v59);
      goto LABEL_14;
    }
    if ( (unsigned int)v59 > (unsigned __int64)*(unsigned int *)(a1 + 204) )
    {
      *(_DWORD *)(a1 + 200) = 0;
      sub_C8D5F0(a1 + 192, (const void *)(a1 + 208), v16, 8u, v13, v14);
      v17 = 0;
      v22 = 8LL * (unsigned int)v59;
      v21 = v58;
      if ( v58 == &v58[v22] )
        goto LABEL_14;
    }
    else
    {
      v19 = v60;
      v20 = 8 * v17;
      v21 = v60;
      if ( *(_DWORD *)(a1 + 200) )
      {
        memmove(*(void **)(a1 + 192), v60, 8 * v17);
        v19 = v58;
        v16 = (unsigned int)v59;
        v17 = v20;
        v21 = &v58[v20];
      }
      v22 = 8 * v16;
      if ( v21 == &v19[v22] )
        goto LABEL_14;
    }
    memcpy((void *)(v17 + *(_QWORD *)(a1 + 192)), v21, v22 - v17);
LABEL_14:
    *(_DWORD *)(a1 + 200) = v18;
    goto LABEL_15;
  }
  v48 = *(_QWORD *)(a1 + 192);
  if ( v48 != a1 + 208 )
  {
    _libc_free(v48);
    v15 = (unsigned __int64)v58;
  }
  *(_QWORD *)(a1 + 192) = v15;
  v49 = v59;
  HIDWORD(v59) = 0;
  *(_QWORD *)(a1 + 200) = v49;
  v58 = v60;
LABEL_15:
  v23 = *(unsigned int *)(a1 + 360);
  v24 = *(_QWORD *)(a1 + 344);
  LODWORD(v59) = 0;
  sub_C7D6A0(v24, 12 * v23, 4);
  v27 = v62;
  ++*(_QWORD *)(a1 + 336);
  v61 += 2;
  *(_QWORD *)(a1 + 344) = v27;
  v62 = 0;
  *(_QWORD *)(a1 + 352) = v63;
  v63 = 0;
  *(_DWORD *)(a1 + 360) = v64;
  v28 = (unsigned __int64)v65;
  v64 = 0;
  if ( v65 == v67 )
  {
    v29 = (unsigned int)v66;
    v30 = *(unsigned int *)(a1 + 376);
    v31 = v66;
    if ( (unsigned int)v66 <= v30 )
    {
      if ( (_DWORD)v66 )
        memmove(*(void **)(a1 + 368), v67, 8LL * (unsigned int)v66);
      goto LABEL_22;
    }
    if ( (unsigned int)v66 > (unsigned __int64)*(unsigned int *)(a1 + 380) )
    {
      *(_DWORD *)(a1 + 376) = 0;
      sub_C8D5F0(a1 + 368, (const void *)(a1 + 384), v29, 8u, v25, v26);
      v30 = 0;
      v35 = 8LL * (unsigned int)v66;
      v34 = v65;
      if ( v65 == &v65[v35] )
        goto LABEL_22;
    }
    else
    {
      v32 = v67;
      v33 = 8 * v30;
      v34 = v67;
      if ( *(_DWORD *)(a1 + 376) )
      {
        memmove(*(void **)(a1 + 368), v67, 8 * v30);
        v32 = v65;
        v29 = (unsigned int)v66;
        v30 = v33;
        v34 = &v65[v33];
      }
      v35 = 8 * v29;
      if ( v34 == &v32[v35] )
        goto LABEL_22;
    }
    memcpy((void *)(v30 + *(_QWORD *)(a1 + 368)), v34, v35 - v30);
LABEL_22:
    *(_DWORD *)(a1 + 376) = v31;
    LODWORD(v66) = 0;
    goto LABEL_23;
  }
  v50 = *(_QWORD *)(a1 + 368);
  if ( v50 != a1 + 384 )
  {
    _libc_free(v50);
    v28 = (unsigned __int64)v65;
  }
  *(_QWORD *)(a1 + 368) = v28;
  v51 = v66;
  v66 = 0;
  *(_QWORD *)(a1 + 376) = v51;
  v65 = v67;
LABEL_23:
  v36 = (_QWORD *)(a1 + 512);
  v37 = *(_QWORD **)(a1 + 512);
  if ( v37 != (_QWORD *)(a1 + 512) )
  {
    do
    {
      v38 = (unsigned __int64)v37;
      v37 = (_QWORD *)*v37;
      v39 = *(_QWORD *)(v38 + 16);
      if ( v39 != v38 + 32 )
        j_j___libc_free_0(v39);
      j_j___libc_free_0(v38);
    }
    while ( v37 != v36 );
  }
  if ( v68 == &v68 )
  {
    *(_QWORD *)(a1 + 520) = v36;
    *(_QWORD *)(a1 + 512) = v36;
    *(_QWORD *)(a1 + 528) = 0;
  }
  else
  {
    *(_QWORD *)(a1 + 512) = v68;
    v40 = v69;
    *(_QWORD *)(a1 + 520) = v69;
    *v40 = v36;
    *(_QWORD *)(*(_QWORD *)(a1 + 512) + 8LL) = v36;
    v69 = &v68;
    *(_QWORD *)(a1 + 528) = v70;
    v68 = &v68;
    v70 = 0;
  }
  *(_DWORD *)(a1 + 536) = v71;
  v57 = &unk_4A0AA50;
  sub_22F3580((__int64)&v57);
  v41 = v68;
  while ( v41 != &v68 )
  {
    v42 = (unsigned __int64)v41;
    v41 = (_QWORD *)*v41;
    v43 = *(_QWORD *)(v42 + 16);
    if ( v43 != v42 + 32 )
      j_j___libc_free_0(v43);
    j_j___libc_free_0(v42);
  }
  if ( v65 != v67 )
    _libc_free((unsigned __int64)v65);
  v57 = &unk_4A08310;
  sub_C7D6A0(v62, 12LL * v64, 4);
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  if ( sub_3097810(a1 + 184, 0x1Fu) || sub_3097810(a1 + 184, 0x1Eu) )
  {
    v44 = sub_3097810(a1 + 184, 0x1Eu);
    sub_3097BA0(a1, v44 != 0, v45, v46, v47);
  }
  sub_3097BF0(a1, a2, a3, a4);
  if ( v72 != v74 )
    _libc_free((unsigned __int64)v72);
}
