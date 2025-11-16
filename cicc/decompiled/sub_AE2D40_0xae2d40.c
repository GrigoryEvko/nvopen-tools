// Function: sub_AE2D40
// Address: 0xae2d40
//
unsigned __int64 *__fastcall sub_AE2D40(unsigned __int64 *a1, __int64 a2, const char *a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  const char **v16; // rdi
  __int32 v17; // r9d
  __int32 v18; // edx
  __int64 v19; // rdx
  __int64 v20; // r13
  char *v21; // rdx
  char *v22; // rsi
  __int64 v23; // rcx
  unsigned int v24; // eax
  __int64 v25; // rdx
  const char **v26; // [rsp-10h] [rbp-F0h]
  __int64 v27; // [rsp-10h] [rbp-F0h]
  __int64 v28; // [rsp-8h] [rbp-E8h]
  unsigned int v29; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v30; // [rsp+12h] [rbp-CEh] BYREF
  unsigned __int8 v31; // [rsp+13h] [rbp-CDh] BYREF
  unsigned int v32; // [rsp+14h] [rbp-CCh] BYREF
  unsigned __int32 v33; // [rsp+18h] [rbp-C8h] BYREF
  unsigned __int32 v34; // [rsp+1Ch] [rbp-C4h] BYREF
  const char *v35; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v36; // [rsp+28h] [rbp-B8h]
  __int64 v37; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v38; // [rsp+38h] [rbp-A8h]
  __int64 v39; // [rsp+40h] [rbp-A0h]
  __int64 *v40; // [rsp+50h] [rbp-90h] BYREF
  __int64 v41; // [rsp+58h] [rbp-88h]
  _BYTE v42[128]; // [rsp+60h] [rbp-80h] BYREF

  v40 = (__int64 *)v42;
  v41 = 0x500000000LL;
  if ( a4 )
  {
    --a4;
    ++a3;
  }
  v35 = a3;
  v6 = (__int64)&v40;
  v36 = a4;
  sub_C93960(&v35, &v40, 58, 0xFFFFFFFFLL, 1);
  if ( (unsigned __int64)(unsigned int)v41 - 3 > 2 )
  {
    LOWORD(v39) = 259;
    v35 = "p[<n>]:<size>:<abi>[:<pref>[:<idx>]]";
    sub_AE1520(
      (__int64)a1,
      (__int64)&v40,
      v7,
      v8,
      v9,
      v10,
      (__int64 *)"p[<n>]:<size>:<abi>[:<pref>[:<idx>]]",
      v36,
      v37,
      v38,
      259);
    goto LABEL_5;
  }
  v12 = v40;
  v32 = 0;
  if ( v40[1] )
  {
    v6 = *v40;
    sub_AE1650(&v35, *v40, v40[1], &v32);
    v13 = (unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL;
    if ( ((unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_18;
    v12 = v40;
  }
  v6 = v12[2];
  sub_AE1770(&v35, v6, v12[3], &v33, (__int64)"pointer size", 12);
  v13 = (unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL;
  if ( ((unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_18;
  v30 = 0;
  v6 = v40[4];
  sub_AE1890(&v35, v6, v40[5], &v30, (__int64)"ABI", 3, 0);
  v16 = v26;
  v13 = (unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL;
  if ( ((unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_18;
  v31 = v30;
  if ( (unsigned int)v41 > 3 )
  {
    v16 = &v35;
    sub_AE1890(&v35, v40[6], v40[7], &v31, (__int64)"preferred", 9, 0);
    v15 = v27;
    v6 = v28;
    v13 = (unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL;
    if ( ((unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_18;
  }
  if ( v30 > v31 )
  {
    v29 = sub_C63BB0(v16, v6, v14, v15);
    v20 = v19;
    v21 = "";
    v35 = (const char *)&v37;
    v22 = "preferred alignment cannot be less than the ABI alignment";
    goto LABEL_20;
  }
  v17 = v33;
  v34 = v33;
  v18 = v33;
  if ( (unsigned int)v41 <= 4 )
  {
LABEL_16:
    v6 = v32;
    sub_AE2A10(a2, v32, v18, v30, v31, v17, 0);
    *a1 = 1;
    goto LABEL_5;
  }
  v6 = v40[8];
  sub_AE1770(&v35, v6, v40[9], &v34, (__int64)"index size", 10);
  v13 = (unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL;
  if ( ((unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
LABEL_18:
    *a1 = v13 | 1;
    goto LABEL_5;
  }
  v17 = v34;
  v18 = v33;
  if ( v34 <= v33 )
    goto LABEL_16;
  v24 = sub_C63BB0(&v35, v6, v33, v23);
  v20 = v25;
  v21 = "";
  v29 = v24;
  v35 = (const char *)&v37;
  v22 = "index size cannot be larger than the pointer size";
LABEL_20:
  sub_AE11D0((__int64 *)&v35, v22, (__int64)v21);
  v6 = (__int64)&v35;
  sub_C63F00(a1, &v35, v29, v20);
  if ( v35 != (const char *)&v37 )
  {
    v6 = v37 + 1;
    j_j___libc_free_0(v35, v37 + 1);
  }
LABEL_5:
  if ( v40 != (__int64 *)v42 )
    _libc_free(v40, v6);
  return a1;
}
