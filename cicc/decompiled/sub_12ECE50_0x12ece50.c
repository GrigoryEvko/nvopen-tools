// Function: sub_12ECE50
// Address: 0x12ece50
//
_QWORD *__fastcall sub_12ECE50(__int64 a1, _DWORD *a2, __int64 *a3, __int64 (__fastcall ***a4)(_QWORD, __int64))
{
  __int64 v5; // rbx
  __int64 v6; // r13
  _BYTE *v7; // rdx
  int v8; // ecx
  _BYTE *v9; // rsi
  __int64 v10; // rax
  _BYTE *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // r13
  _QWORD *v21; // rbx
  _QWORD *v22; // r14
  _QWORD *v23; // rdi
  _QWORD *v24; // rax
  _QWORD *v25; // r14
  _QWORD *v26; // r13
  _QWORD *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  _QWORD *result; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  int v34; // ebx
  _BYTE *v35; // rcx
  __int64 v36; // r13
  __int64 v37; // rdx
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  int v40; // ebx
  _BYTE *v41; // rcx
  __int64 v42; // r13
  __int64 v43; // rdx
  __int64 v44; // [rsp-8h] [rbp-608h]
  char v48; // [rsp+48h] [rbp-5B8h] BYREF
  char v49; // [rsp+4Ch] [rbp-5B4h] BYREF
  void *v50; // [rsp+50h] [rbp-5B0h] BYREF
  _BYTE *v51; // [rsp+58h] [rbp-5A8h]
  __int64 v52; // [rsp+60h] [rbp-5A0h]
  _BYTE v53[128]; // [rsp+68h] [rbp-598h] BYREF
  __int64 v54; // [rsp+E8h] [rbp-518h]
  __int64 v55; // [rsp+F0h] [rbp-510h]
  __int64 v56; // [rsp+F8h] [rbp-508h]
  int v57; // [rsp+100h] [rbp-500h]
  _BYTE *v58; // [rsp+108h] [rbp-4F8h]
  __int64 v59; // [rsp+110h] [rbp-4F0h]
  _BYTE v60[128]; // [rsp+118h] [rbp-4E8h] BYREF
  _QWORD *v61; // [rsp+198h] [rbp-468h] BYREF
  _QWORD **v62; // [rsp+1A0h] [rbp-460h]
  __int64 v63; // [rsp+1A8h] [rbp-458h]
  int v64; // [rsp+1B0h] [rbp-450h]
  _BYTE *v65; // [rsp+1C0h] [rbp-440h] BYREF
  __int64 v66; // [rsp+1C8h] [rbp-438h]
  _BYTE v67[1072]; // [rsp+1D0h] [rbp-430h] BYREF

  v5 = (int)*a2;
  v65 = v67;
  v6 = *a3;
  v66 = 0x8000000000LL;
  if ( (unsigned __int64)(8 * v5) > 0x400 )
  {
    sub_16CD150(&v65, v67, v5, 8);
    LODWORD(v7) = (_DWORD)v65;
    v8 = v66;
    v9 = &v65[8 * (unsigned int)v66];
  }
  else
  {
    v7 = v67;
    v8 = 0;
    v9 = v67;
  }
  if ( 8 * v5 > 0 )
  {
    v10 = 0;
    do
    {
      *(_QWORD *)&v9[8 * v10] = *(_QWORD *)(v6 + 8 * v10);
      ++v10;
    }
    while ( v5 - v10 > 0 );
    LODWORD(v7) = (_DWORD)v65;
    v8 = v66;
  }
  LODWORD(v66) = v8 + v5;
  sub_1691F20((unsigned int)&v50, a1 + 8, (_DWORD)v7, v8 + v5, (unsigned int)&v48, (unsigned int)&v49, 0, 0);
  sub_168FB40(a1 + 120);
  v11 = v51;
  v12 = v44;
  if ( v51 == v53 )
  {
    v38 = (unsigned int)v52;
    v39 = *(unsigned int *)(a1 + 136);
    v40 = v52;
    if ( (unsigned int)v52 <= v39 )
    {
      if ( (_DWORD)v52 )
      {
        v12 = (__int64)v53;
        memmove(*(void **)(a1 + 128), v53, 8LL * (unsigned int)v52);
      }
    }
    else
    {
      if ( (unsigned int)v52 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
      {
        *(_DWORD *)(a1 + 136) = 0;
        sub_16CD150(a1 + 128, a1 + 144, v38, 8);
        v41 = v51;
        v38 = (unsigned int)v52;
        v39 = 0;
        v12 = (__int64)v51;
      }
      else
      {
        v41 = v53;
        v42 = 8 * v39;
        v12 = (__int64)v53;
        if ( *(_DWORD *)(a1 + 136) )
        {
          memmove(*(void **)(a1 + 128), v53, 8 * v39);
          v41 = v51;
          v38 = (unsigned int)v52;
          v39 = v42;
          v12 = (__int64)&v51[v42];
        }
      }
      v43 = 8 * v38;
      if ( (_BYTE *)v12 != &v41[v43] )
        memcpy((void *)(v39 + *(_QWORD *)(a1 + 128)), (const void *)v12, v43 - v39);
    }
    *(_DWORD *)(a1 + 136) = v40;
  }
  else
  {
    v13 = *(_QWORD *)(a1 + 128);
    if ( v13 != a1 + 144 )
    {
      _libc_free(v13, v44);
      v11 = v51;
    }
    *(_QWORD *)(a1 + 128) = v11;
    v14 = v52;
    HIDWORD(v52) = 0;
    *(_QWORD *)(a1 + 136) = v14;
    v51 = v53;
  }
  v15 = *(_QWORD *)(a1 + 280);
  LODWORD(v52) = 0;
  j___libc_free_0(v15);
  v16 = v55;
  ++*(_QWORD *)(a1 + 272);
  v54 += 2;
  *(_QWORD *)(a1 + 280) = v16;
  v55 = 0;
  *(_QWORD *)(a1 + 288) = v56;
  v56 = 0;
  *(_DWORD *)(a1 + 296) = v57;
  v17 = v58;
  v57 = 0;
  if ( v58 == v60 )
  {
    v32 = (unsigned int)v59;
    v33 = *(unsigned int *)(a1 + 312);
    v34 = v59;
    if ( (unsigned int)v59 <= v33 )
    {
      if ( (_DWORD)v59 )
      {
        v12 = (__int64)v60;
        memmove(*(void **)(a1 + 304), v60, 8LL * (unsigned int)v59);
      }
    }
    else
    {
      if ( (unsigned int)v59 > (unsigned __int64)*(unsigned int *)(a1 + 316) )
      {
        *(_DWORD *)(a1 + 312) = 0;
        sub_16CD150(a1 + 304, a1 + 320, v32, 8);
        v35 = v58;
        v32 = (unsigned int)v59;
        v33 = 0;
        v12 = (__int64)v58;
      }
      else
      {
        v35 = v60;
        v36 = 8 * v33;
        v12 = (__int64)v60;
        if ( *(_DWORD *)(a1 + 312) )
        {
          memmove(*(void **)(a1 + 304), v60, 8 * v33);
          v35 = v58;
          v32 = (unsigned int)v59;
          v33 = v36;
          v12 = (__int64)&v58[v36];
        }
      }
      v37 = 8 * v32;
      if ( (_BYTE *)v12 != &v35[v37] )
        memcpy((void *)(v33 + *(_QWORD *)(a1 + 304)), (const void *)v12, v37 - v33);
    }
    *(_DWORD *)(a1 + 312) = v34;
    LODWORD(v59) = 0;
  }
  else
  {
    v18 = *(_QWORD *)(a1 + 304);
    if ( v18 != a1 + 320 )
    {
      _libc_free(v18, v12);
      v17 = v58;
    }
    *(_QWORD *)(a1 + 304) = v17;
    v19 = v59;
    v59 = 0;
    *(_QWORD *)(a1 + 312) = v19;
    v58 = v60;
  }
  v20 = (_QWORD *)(a1 + 448);
  v21 = *(_QWORD **)(a1 + 448);
  if ( v21 != (_QWORD *)(a1 + 448) )
  {
    do
    {
      v22 = v21;
      v21 = (_QWORD *)*v21;
      v23 = (_QWORD *)v22[2];
      if ( v23 != v22 + 4 )
        j_j___libc_free_0(v23, v22[4] + 1LL);
      v12 = 48;
      j_j___libc_free_0(v22, 48);
    }
    while ( v21 != v20 );
  }
  if ( v61 == &v61 )
  {
    *(_QWORD *)(a1 + 456) = v20;
    *(_QWORD *)(a1 + 448) = v20;
    *(_QWORD *)(a1 + 464) = 0;
  }
  else
  {
    *(_QWORD *)(a1 + 448) = v61;
    v24 = v62;
    *(_QWORD *)(a1 + 456) = v62;
    *v24 = v20;
    *(_QWORD *)(*(_QWORD *)(a1 + 448) + 8LL) = v20;
    v62 = &v61;
    *(_QWORD *)(a1 + 464) = v63;
    v61 = &v61;
    v63 = 0;
  }
  *(_DWORD *)(a1 + 472) = v64;
  v50 = &unk_49EE9E8;
  sub_168FB40(&v50);
  v25 = v61;
  while ( v25 != &v61 )
  {
    v26 = v25;
    v25 = (_QWORD *)*v25;
    v27 = (_QWORD *)v26[2];
    if ( v27 != v26 + 4 )
      j_j___libc_free_0(v27, v26[4] + 1LL);
    v12 = 48;
    j_j___libc_free_0(v26, 48);
  }
  if ( v58 != v60 )
    _libc_free(v58, v12);
  v50 = &unk_49E6A18;
  j___libc_free_0(v55);
  if ( v51 != v53 )
    _libc_free(v51, v12);
  if ( sub_12EC800(a1 + 120, 0xA1u) || sub_12EC800(a1 + 120, 0xA0u) )
  {
    v28 = sub_12EC800(a1 + 120, 0xA0u);
    sub_12ECA50(a1, v28 != 0, v29, v30);
  }
  result = sub_12ECC30(a1, a2, a3, a4);
  if ( v65 != v67 )
    return (_QWORD *)_libc_free(v65, a2);
  return result;
}
