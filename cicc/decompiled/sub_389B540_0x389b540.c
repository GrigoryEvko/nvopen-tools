// Function: sub_389B540
// Address: 0x389b540
//
__int64 __fastcall sub_389B540(
        __int64 *a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int16 *v10; // r12
  int v11; // eax
  __int64 v12; // r12
  __int64 v13; // r14
  _QWORD *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // r13
  __int64 *v18; // rsi
  __int64 v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // r12
  unsigned __int64 v23; // rdx
  __int64 v24; // rdi
  unsigned int v25; // r12d
  int *v26; // r13
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v32; // r13
  __int64 v33; // rsi
  __int64 v34; // rbx
  __int64 v35; // rdi
  char *v36; // rax
  __int64 v37; // rdx
  _BYTE *v38; // rdi
  char *v39; // rax
  __int64 v40; // rdi
  size_t v41; // rdx
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 i; // rbx
  int *v45; // [rsp+0h] [rbp-120h]
  void *v46; // [rsp+28h] [rbp-F8h]
  char *v47; // [rsp+30h] [rbp-F0h] BYREF
  size_t n; // [rsp+38h] [rbp-E8h]
  _QWORD src[2]; // [rsp+40h] [rbp-E0h] BYREF
  int v50; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v51; // [rsp+58h] [rbp-C8h]
  int v52; // [rsp+60h] [rbp-C0h]
  __int64 v53; // [rsp+68h] [rbp-B8h]
  void *dest; // [rsp+70h] [rbp-B0h]
  size_t v55; // [rsp+78h] [rbp-A8h]
  _QWORD v56[2]; // [rsp+80h] [rbp-A0h] BYREF
  _QWORD *v57; // [rsp+90h] [rbp-90h]
  __int64 v58; // [rsp+98h] [rbp-88h]
  _BYTE v59[16]; // [rsp+A0h] [rbp-80h] BYREF
  unsigned __int64 v60; // [rsp+B0h] [rbp-70h]
  unsigned int v61; // [rsp+B8h] [rbp-68h]
  char v62; // [rsp+BCh] [rbp-64h]
  void *v63; // [rsp+C8h] [rbp-58h] BYREF
  __int64 v64; // [rsp+D0h] [rbp-50h]
  unsigned __int64 v65; // [rsp+E8h] [rbp-38h]

  dest = v56;
  v50 = 0;
  v51 = 0;
  v53 = 0;
  v55 = 0;
  LOBYTE(v56[0]) = 0;
  v57 = v59;
  v58 = 0;
  v59[0] = 0;
  v61 = 1;
  v60 = 0;
  v62 = 0;
  v10 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)&v47, 0.0);
  sub_169E320(&v63, (__int64 *)&v47, v10);
  sub_1698460((__int64)&v47);
  v11 = *((_DWORD *)a1 + 34);
  v65 = 0;
  if ( v11 != -1 )
  {
    v50 = 1;
    v52 = v11;
    goto LABEL_3;
  }
  v35 = a1[1];
  v50 = 3;
  v36 = (char *)sub_1649960(v35);
  v47 = (char *)src;
  if ( !v36 )
  {
    LOBYTE(src[0]) = 0;
    v38 = dest;
    v41 = 0;
LABEL_48:
    v55 = v41;
    v38[v41] = 0;
    v39 = v47;
    goto LABEL_45;
  }
  sub_3887410((__int64 *)&v47, v36, (__int64)&v36[v37]);
  v38 = dest;
  v39 = (char *)dest;
  if ( v47 == (char *)src )
  {
    v41 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v41 = n;
      v38 = dest;
    }
    goto LABEL_48;
  }
  if ( dest == v56 )
  {
    dest = v47;
    v55 = n;
    v56[0] = src[0];
  }
  else
  {
    v40 = v56[0];
    dest = v47;
    v55 = n;
    v56[0] = src[0];
    if ( v39 )
    {
      v47 = v39;
      src[0] = v40;
      goto LABEL_45;
    }
  }
  v47 = (char *)src;
  v39 = (char *)src;
LABEL_45:
  n = 0;
  *v39 = 0;
  if ( v47 != (char *)src )
    j_j___libc_free_0((unsigned __int64)v47);
LABEL_3:
  v12 = *a1;
  v13 = sub_38911E0(*a1 + 1072, (__int64)&v50);
  v45 = (int *)v13;
  v14 = (_QWORD *)(v12 + 1080);
  v46 = sub_16982C0();
  v15 = v13;
  if ( v13 == v12 + 1080 )
  {
    v25 = 0;
  }
  else
  {
    v16 = *(_QWORD *)(v13 + 216);
    v17 = v15 + 200;
    if ( v15 + 200 != v16 )
    {
      do
      {
        v22 = *(_QWORD *)(v16 + 192);
        v23 = *(_QWORD *)(v16 + 40);
        if ( *(_DWORD *)(v16 + 32) == 2 )
        {
          v18 = sub_389AAD0(a1, v16 + 64, v23);
          if ( !v18 )
          {
LABEL_10:
            v24 = *a1;
            v47 = "referenced value is not a basic block";
            LOWORD(src[0]) = 259;
            v25 = sub_38814C0(v24 + 8, *(_QWORD *)(v16 + 40), (__int64)&v47);
            goto LABEL_24;
          }
        }
        else
        {
          v18 = sub_389B190(a1, *(_DWORD *)(v16 + 48), v23);
          if ( !v18 )
            goto LABEL_10;
        }
        v19 = sub_159BBF0(a1[1], (__int64)v18);
        sub_164D160(v22, v19, (__m128)0LL, a3, a4, a5, v20, v21, a8, a9);
        sub_15E5B20(v22);
        v16 = sub_220EEE0(v16);
      }
      while ( v17 != v16 );
      v12 = *a1;
      v14 = (_QWORD *)(*a1 + 1080);
    }
    v26 = sub_220F330(v45, v14);
    sub_3889CD0(*((_QWORD *)v26 + 26));
    v27 = *((_QWORD *)v26 + 23);
    if ( v27 )
      j_j___libc_free_0_0(v27);
    if ( *((void **)v26 + 19) == v46 )
    {
      v42 = *((_QWORD *)v26 + 20);
      if ( v42 )
      {
        v43 = 32LL * *(_QWORD *)(v42 - 8);
        for ( i = v42 + v43; v42 != i; sub_127D120((_QWORD *)(i + 8)) )
          i -= 32;
        j_j_j___libc_free_0_0(v42 - 8);
      }
    }
    else
    {
      sub_1698460((__int64)(v26 + 38));
    }
    if ( (unsigned int)v26[34] > 0x40 )
    {
      v28 = *((_QWORD *)v26 + 16);
      if ( v28 )
        j_j___libc_free_0_0(v28);
    }
    v29 = *((_QWORD *)v26 + 12);
    if ( (int *)v29 != v26 + 28 )
      j_j___libc_free_0(v29);
    v30 = *((_QWORD *)v26 + 8);
    if ( (int *)v30 != v26 + 20 )
      j_j___libc_free_0(v30);
    j_j___libc_free_0((unsigned __int64)v26);
    --*(_QWORD *)(v12 + 1112);
    v25 = 0;
  }
LABEL_24:
  if ( v65 )
    j_j___libc_free_0_0(v65);
  if ( v63 == v46 )
  {
    v32 = v64;
    if ( v64 )
    {
      v33 = 32LL * *(_QWORD *)(v64 - 8);
      v34 = v64 + v33;
      if ( v64 != v64 + v33 )
      {
        do
        {
          v34 -= 32;
          sub_127D120((_QWORD *)(v34 + 8));
        }
        while ( v32 != v34 );
      }
      j_j_j___libc_free_0_0(v32 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v63);
  }
  if ( v61 > 0x40 && v60 )
    j_j___libc_free_0_0(v60);
  if ( v57 != (_QWORD *)v59 )
    j_j___libc_free_0((unsigned __int64)v57);
  if ( dest != v56 )
    j_j___libc_free_0((unsigned __int64)dest);
  return v25;
}
