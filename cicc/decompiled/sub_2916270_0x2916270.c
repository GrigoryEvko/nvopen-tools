// Function: sub_2916270
// Address: 0x2916270
//
__int64 __fastcall sub_2916270(
        _QWORD *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6)
{
  __int64 *v6; // r14
  __int64 *v7; // rdx
  __int64 *v9; // r13
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // r13
  __int64 v13; // r12
  __int64 v14; // rdx
  unsigned __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17; // rax
  char *v18; // r13
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 *v21; // rdi
  int v22; // eax
  __int64 v23; // r9
  char *v24; // r10
  __int64 v25; // rdx
  char *v26; // rax
  size_t v27; // r11
  __int64 v28; // r13
  unsigned __int64 v29; // r8
  __int64 v31; // rax
  __int64 v32; // rax
  size_t v33; // [rsp+0h] [rbp-D0h]
  char *v34; // [rsp+8h] [rbp-C8h]
  char *v35; // [rsp+10h] [rbp-C0h]
  char v38; // [rsp+2Fh] [rbp-A1h]
  char v39; // [rsp+30h] [rbp-A0h]
  __int64 v40; // [rsp+30h] [rbp-A0h]
  void *src; // [rsp+48h] [rbp-88h] BYREF
  __int64 *v42; // [rsp+50h] [rbp-80h] BYREF
  __int64 v43; // [rsp+58h] [rbp-78h]
  _BYTE v44[112]; // [rsp+60h] [rbp-70h] BYREF

  v6 = (__int64 *)a1[3];
  v7 = (__int64 *)a1[2];
  v42 = (__int64 *)v44;
  v43 = 0x800000000LL;
  src = v7;
  if ( v7 == v6 )
  {
    v20 = 0;
LABEL_32:
    v42[v20] = 4096;
    LODWORD(v43) = v43 + 1;
    v31 = (unsigned int)v43;
    if ( (unsigned __int64)(unsigned int)v43 + 1 > HIDWORD(v43) )
    {
      sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 8u, a5, a6);
      v31 = (unsigned int)v43;
    }
    v42[v31] = a3;
    LODWORD(v43) = v43 + 1;
    v32 = (unsigned int)v43;
    if ( (unsigned __int64)(unsigned int)v43 + 1 > HIDWORD(v43) )
    {
      sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 8u, a5, a6);
      v32 = (unsigned int)v43;
    }
    v42[v32] = a2;
    v20 = (unsigned int)(v43 + 1);
    LODWORD(v43) = v43 + 1;
LABEL_18:
    v21 = (__int64 *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
    if ( (a1[1] & 4) != 0 )
      v21 = (__int64 *)*v21;
    v19 = sub_B0D000(v21, v42, v20, 0, 1);
    goto LABEL_27;
  }
  v39 = 0;
  v9 = v7;
  v38 = 0;
  do
  {
    v10 = *v9;
    if ( *v9 == 4096 )
    {
      v38 = 1;
    }
    else if ( (unsigned __int64)(v10 - 4102) > 1 )
    {
      v22 = sub_AF4160((unsigned __int64 **)&src);
      v24 = (char *)src;
      v25 = (unsigned int)v43;
      v26 = (char *)&v9[v22];
      v27 = v26 - (_BYTE *)src;
      v28 = (v26 - (_BYTE *)src) >> 3;
      v29 = v28 + (unsigned int)v43;
      if ( v29 > HIDWORD(v43) )
      {
        v33 = v26 - (_BYTE *)src;
        v34 = (char *)src;
        v35 = v26;
        sub_C8D5F0((__int64)&v42, v44, v28 + (unsigned int)v43, 8u, v29, v23);
        v25 = (unsigned int)v43;
        v27 = v33;
        v24 = v34;
        v26 = v35;
      }
      if ( v26 != v24 )
      {
        memcpy(&v42[v25], v24, v27);
        LODWORD(v25) = v43;
      }
      LODWORD(v43) = v28 + v25;
    }
    else
    {
      v11 = v9[1];
      v12 = v9[2];
      if ( a2 < v12 || (v13 = a4 + v11, a4 + v11 < 0) )
      {
        v19 = 0;
        goto LABEL_27;
      }
      v14 = (unsigned int)v43;
      v15 = (unsigned int)v43 + 1LL;
      if ( v15 > HIDWORD(v43) )
      {
        v40 = v10;
        sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 8u, a5, v15);
        v14 = (unsigned int)v43;
        v10 = v40;
      }
      v42[v14] = v10;
      LODWORD(v43) = v43 + 1;
      v16 = (unsigned int)v43;
      if ( (unsigned __int64)(unsigned int)v43 + 1 > HIDWORD(v43) )
      {
        sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 8u, a5, v15);
        v16 = (unsigned int)v43;
      }
      v42[v16] = v13;
      LODWORD(v43) = v43 + 1;
      v17 = (unsigned int)v43;
      if ( (unsigned __int64)(unsigned int)v43 + 1 > HIDWORD(v43) )
      {
        sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 8u, a5, v15);
        v17 = (unsigned int)v43;
      }
      v39 = 1;
      v42[v17] = v12;
      LODWORD(v43) = v43 + 1;
    }
    v18 = (char *)src;
    v9 = (__int64 *)&v18[8 * (unsigned int)sub_AF4160((unsigned __int64 **)&src)];
    src = v9;
  }
  while ( v6 != v9 );
  if ( !v38 || (v19 = 0, !v39) )
  {
    v20 = (unsigned int)v43;
    if ( v39 )
      goto LABEL_18;
    a5 = (unsigned int)v43 + 1LL;
    if ( a5 > HIDWORD(v43) )
    {
      sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 8u, a5, a6);
      v20 = (unsigned int)v43;
    }
    goto LABEL_32;
  }
LABEL_27:
  if ( v42 != (__int64 *)v44 )
    _libc_free((unsigned __int64)v42);
  return v19;
}
