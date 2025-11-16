// Function: sub_B1D3D0
// Address: 0xb1d3d0
//
__int64 *__fastcall sub_B1D3D0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rbx
  __int64 *v7; // r13
  __int64 v8; // rcx
  unsigned int v9; // r15d
  _BYTE *v10; // r12
  __int64 v11; // r14
  unsigned __int64 v12; // rdx
  int v13; // eax
  int v14; // ecx
  _QWORD *v15; // rbx
  int v16; // r12d
  __int64 v17; // rsi
  char v18; // cl
  __int64 v19; // rdi
  int v20; // eax
  unsigned int v21; // edx
  __int64 *v22; // r15
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 *v26; // r12
  __int64 *i; // r14
  __int64 v28; // rsi
  int v29; // r12d
  __int64 v30; // rsi
  int v31; // r12d
  __int64 v32; // rsi
  int v33; // r12d
  __int64 v34; // rsi
  int v35; // r12d
  __int64 *result; // rax
  __int64 v37; // rdx
  unsigned int v38; // r12d
  int v39; // r15d
  _BYTE *v40; // r14
  int v41; // eax
  _QWORD *v42; // rbx
  __int64 v43; // r14
  int v44; // r9d
  __int64 *v45; // rbx
  char v46; // r8
  __int64 *v47; // rbx
  char v48; // r8
  __int64 *v49; // rbx
  bool v50; // zf
  __int64 *v52; // [rsp+10h] [rbp-F0h]
  int v53; // [rsp+28h] [rbp-D8h]
  _QWORD *v54; // [rsp+28h] [rbp-D8h]
  _QWORD *v55; // [rsp+28h] [rbp-D8h]
  int v56; // [rsp+30h] [rbp-D0h]
  __int64 v57; // [rsp+30h] [rbp-D0h]
  __int64 v58; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v59; // [rsp+38h] [rbp-C8h]
  __int64 v61; // [rsp+58h] [rbp-A8h] BYREF
  char v62[8]; // [rsp+60h] [rbp-A0h] BYREF
  int v63; // [rsp+68h] [rbp-98h]
  __int64 v64; // [rsp+70h] [rbp-90h]
  unsigned int v65; // [rsp+78h] [rbp-88h]
  _BYTE *v66; // [rsp+80h] [rbp-80h] BYREF
  __int64 v67; // [rsp+88h] [rbp-78h]
  _BYTE v68[112]; // [rsp+90h] [rbp-70h] BYREF

  v3 = a1;
  v4 = (a2 - (__int64)a1) >> 5;
  v5 = (a2 - (__int64)a1) >> 3;
  if ( v4 > 0 )
  {
    v6 = &v66;
    v7 = a1;
    v52 = &a1[4 * v4];
    while ( !a3 )
    {
      v61 = *v7;
      sub_B1C7C0((__int64)v62, &v61);
      v38 = v65;
      v39 = v63;
      v40 = v68;
      v57 = v64;
      v67 = 0x800000000LL;
      v66 = v68;
      v41 = 0;
      v59 = (int)(v65 - v63);
      if ( v59 > 8 )
      {
        sub_C8D5F0(v6, v68, v59, 8);
        v41 = v67;
        v40 = &v66[8 * (unsigned int)v67];
      }
      if ( v39 != v38 )
      {
        v55 = v6;
        v42 = v40;
        do
        {
          --v38;
          if ( v42 )
            *v42 = sub_B46EC0(v57, v38);
          ++v42;
        }
        while ( v39 != v38 );
        v6 = v55;
        v41 = v67;
      }
      v17 = 0;
      LODWORD(v67) = v59 + v41;
      sub_B1C8F0((__int64)v6);
LABEL_21:
      v29 = v67;
      if ( v66 != v68 )
        _libc_free(v66, v17);
      if ( v29 )
        return v7;
      v30 = v7[1];
      sub_B1CB80(v6, v30, a3);
      v31 = v67;
      if ( v66 != v68 )
        _libc_free(v66, v30);
      if ( v31 )
        return v7 + 1;
      v32 = v7[2];
      sub_B1CB80(v6, v32, a3);
      v33 = v67;
      if ( v66 != v68 )
        _libc_free(v66, v32);
      if ( v33 )
        return v7 + 2;
      v34 = v7[3];
      sub_B1CB80(v6, v34, a3);
      v35 = v67;
      if ( v66 != v68 )
        _libc_free(v66, v34);
      if ( v35 )
        return v7 + 3;
      v7 += 4;
      if ( v52 == v7 )
      {
        v3 = v7;
        v5 = (a2 - (__int64)v7) >> 3;
        goto LABEL_35;
      }
    }
    v8 = *(_QWORD *)(a3 + 8);
    v61 = *v7;
    v58 = v8;
    sub_B1C7C0((__int64)v62, &v61);
    v9 = v65;
    v67 = 0x800000000LL;
    v10 = v68;
    v53 = v63;
    v11 = v64;
    v66 = v68;
    v12 = (int)(v65 - v63);
    v13 = 0;
    v56 = v65 - v63;
    if ( v12 > 8 )
    {
      sub_C8D5F0(v6, v68, v12, 8);
      v13 = v67;
      v10 = &v66[8 * (unsigned int)v67];
    }
    v14 = v53;
    if ( v53 != v9 )
    {
      v54 = v6;
      v15 = v10;
      v16 = v14;
      do
      {
        --v9;
        if ( v15 )
          *v15 = sub_B46EC0(v11, v9);
        ++v15;
      }
      while ( v16 != v9 );
      v6 = v54;
      v13 = v67;
    }
    LODWORD(v67) = v56 + v13;
    sub_B1C8F0((__int64)v6);
    v17 = v61;
    v18 = *(_BYTE *)(v58 + 312) & 1;
    if ( v18 )
    {
      v19 = v58 + 320;
      v20 = 3;
    }
    else
    {
      v37 = *(unsigned int *)(v58 + 328);
      v19 = *(_QWORD *)(v58 + 320);
      if ( !(_DWORD)v37 )
        goto LABEL_51;
      v20 = v37 - 1;
    }
    v21 = v20 & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
    v22 = (__int64 *)(v19 + 72LL * v21);
    v23 = *v22;
    if ( v61 == *v22 )
    {
LABEL_15:
      v24 = 288;
      if ( !v18 )
        v24 = 72LL * *(unsigned int *)(v58 + 328);
      if ( v22 != (__int64 *)(v19 + v24) )
      {
        v25 = (__int64 *)v22[1];
        v26 = &v25[*((unsigned int *)v22 + 4)];
        for ( i = v25; v26 != i; ++i )
        {
          v28 = *i;
          sub_B1CA60((__int64)v6, v28);
        }
        v17 = (__int64)(v22 + 5);
        sub_B1CB00((__int64)v6, (__int64)(v22 + 5));
      }
      goto LABEL_21;
    }
    v44 = 1;
    while ( v23 != -4096 )
    {
      v21 = v20 & (v44 + v21);
      v22 = (__int64 *)(v19 + 72LL * v21);
      v23 = *v22;
      if ( v61 == *v22 )
        goto LABEL_15;
      ++v44;
    }
    if ( v18 )
    {
      v43 = 288;
      goto LABEL_52;
    }
    v37 = *(unsigned int *)(v58 + 328);
LABEL_51:
    v43 = 72 * v37;
LABEL_52:
    v22 = (__int64 *)(v19 + v43);
    goto LABEL_15;
  }
LABEL_35:
  if ( v5 != 2 )
  {
    if ( v5 != 3 )
    {
      if ( v5 != 1 )
        return (__int64 *)a2;
      goto LABEL_63;
    }
    v45 = v3;
    v46 = sub_B1CE10(*v3, a3);
    result = v45;
    if ( v46 )
      return result;
    v3 = v45 + 1;
  }
  v47 = v3;
  v48 = sub_B1CE10(*v3, a3);
  result = v47;
  if ( v48 )
    return result;
  v3 = v47 + 1;
LABEL_63:
  v49 = v3;
  v50 = (unsigned __int8)sub_B1CE10(*v3, a3) == 0;
  result = v49;
  if ( v50 )
    return (__int64 *)a2;
  return result;
}
