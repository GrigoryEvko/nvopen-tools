// Function: sub_185C140
// Address: 0x185c140
//
void __fastcall sub_185C140(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *v11; // rax
  __int64 *v12; // rbx
  __int64 ****v13; // rax
  __int64 v14; // rdx
  __int64 ****v15; // r14
  __int64 ***v16; // rdi
  __int64 ****v17; // r15
  size_t v18; // rsi
  __int64 *v19; // r14
  __int128 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // r15
  _QWORD *v23; // r13
  double v24; // xmm4_8
  double v25; // xmm5_8
  int v26; // r8d
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 ****v29; // rax
  __int64 v30; // [rsp+0h] [rbp-B0h]
  __int64 v31; // [rsp+0h] [rbp-B0h]
  _BYTE v32[16]; // [rsp+10h] [rbp-A0h] BYREF
  __int16 v33; // [rsp+20h] [rbp-90h]
  void *base; // [rsp+30h] [rbp-80h] BYREF
  __int64 v35; // [rsp+38h] [rbp-78h]
  _BYTE v36[112]; // [rsp+40h] [rbp-70h] BYREF

  if ( *(_DWORD *)(a2 + 28) == *(_DWORD *)(a2 + 32) )
  {
    sub_15E55B0(a1);
  }
  else
  {
    v11 = (_QWORD *)sub_16498A0(a1);
    v12 = (__int64 *)sub_16471D0(v11, 0);
    base = v36;
    v35 = 0x800000000LL;
    v13 = *(__int64 *****)(a2 + 16);
    if ( v13 == *(__int64 *****)(a2 + 8) )
      v14 = *(unsigned int *)(a2 + 28);
    else
      v14 = *(unsigned int *)(a2 + 24);
    v15 = &v13[v14];
    if ( v13 == v15 )
      goto LABEL_7;
    while ( 1 )
    {
      v16 = *v13;
      v17 = v13;
      if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v15 == ++v13 )
        goto LABEL_7;
    }
    if ( v15 == v13 )
    {
LABEL_7:
      v18 = 0;
    }
    else
    {
      do
      {
        v27 = sub_15A4AD0(v16, (__int64)v12);
        v28 = (unsigned int)v35;
        if ( (unsigned int)v35 >= HIDWORD(v35) )
        {
          v31 = v27;
          sub_16CD150((__int64)&base, v36, 0, 8, v26, v27);
          v28 = (unsigned int)v35;
          v27 = v31;
        }
        *((_QWORD *)base + v28) = v27;
        v18 = (unsigned int)(v35 + 1);
        v29 = v17 + 1;
        LODWORD(v35) = v35 + 1;
        if ( v17 + 1 == v15 )
          break;
        while ( 1 )
        {
          v16 = *v29;
          v17 = v29;
          if ( (unsigned __int64)*v29 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v15 == ++v29 )
            goto LABEL_19;
        }
      }
      while ( v15 != v29 );
LABEL_19:
      if ( 8 * v18 > 8 )
      {
        qsort(base, v18, 8u, (__compar_fn_t)sub_185AEF0);
        v18 = (unsigned int)v35;
      }
    }
    v19 = sub_1645D80(v12, v18);
    v30 = *(_QWORD *)(a1 + 40);
    sub_15E53F0((_QWORD *)a1);
    *((_QWORD *)&v20 + 1) = base;
    *(_QWORD *)&v20 = v19;
    v22 = sub_159DFD0(v20, (unsigned int)v35, v21);
    v33 = 257;
    v23 = sub_1648A60(88, 1u);
    if ( v23 )
      sub_15E51E0((__int64)v23, v30, (__int64)v19, 0, 6, v22, (__int64)v32, 0, 0, 0, 0);
    sub_164B7C0((__int64)v23, a1);
    sub_15E5D20((__int64)v23, "llvm.metadata", 0xDu);
    sub_15E5530(a1);
    sub_159D9E0(a1);
    sub_164BE60(a1, a3, a4, a5, a6, v24, v25, a9, a10);
    *(_DWORD *)(a1 + 20) = *(_DWORD *)(a1 + 20) & 0xF0000000 | 1;
    sub_1648B90(a1);
    if ( base != v36 )
      _libc_free((unsigned __int64)base);
  }
}
