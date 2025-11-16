// Function: sub_AE2680
// Address: 0xae2680
//
unsigned __int64 *__fastcall sub_AE2680(unsigned __int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  char v5; // r14
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r13
  char *v17; // rdx
  char *v18; // rsi
  unsigned int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // [rsp-10h] [rbp-D0h]
  __int64 v22; // [rsp-8h] [rbp-C8h]
  unsigned int v23; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v24; // [rsp+1Ah] [rbp-A6h] BYREF
  unsigned __int8 v25; // [rsp+1Bh] [rbp-A5h] BYREF
  unsigned int v26; // [rsp+1Ch] [rbp-A4h] BYREF
  __int64 *v27; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v28; // [rsp+28h] [rbp-98h]
  const char *v29; // [rsp+30h] [rbp-90h] BYREF
  __int64 v30; // [rsp+38h] [rbp-88h]
  __int64 v31; // [rsp+40h] [rbp-80h]
  __int64 *v32; // [rsp+50h] [rbp-70h] BYREF
  __int64 v33; // [rsp+58h] [rbp-68h]
  _BYTE v34[96]; // [rsp+60h] [rbp-60h] BYREF

  v32 = (__int64 *)v34;
  v5 = *(_BYTE *)a3;
  v33 = 0x300000000LL;
  if ( a4 )
  {
    --a4;
    a3 = (__int64 *)((char *)a3 + 1);
  }
  v27 = a3;
  v6 = (__int64)&v32;
  v28 = a4;
  sub_C93960(&v27, &v32, 58, 0xFFFFFFFFLL, 1);
  if ( (unsigned __int64)(unsigned int)v33 - 2 > 1 )
  {
    LOBYTE(v27) = v5;
    LOWORD(v31) = 776;
    v29 = "<size>:<abi>[:<pref>]";
    sub_AE1520((__int64)a1, (__int64)&v32, v7, v8, v9, v10, v27, v28, (int)"<size>:<abi>[:<pref>]", v30, 776);
    goto LABEL_5;
  }
  v6 = *v32;
  sub_AE1770(&v27, *v32, v32[1], &v26, (__int64)"size", 4);
  v12 = (unsigned __int64)v27 & 0xFFFFFFFFFFFFFFFELL;
  if ( ((unsigned __int64)v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_17;
  v24 = 0;
  sub_AE1890(&v27, v32[2], v32[3], &v24, (__int64)"ABI", 3, 0);
  v14 = v21;
  v6 = v22;
  v12 = (unsigned __int64)v27 & 0xFFFFFFFFFFFFFFFELL;
  if ( ((unsigned __int64)v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_17;
  if ( v5 == 105 )
  {
    v13 = v26;
    if ( v26 == 8 )
    {
      if ( v24 )
      {
        v19 = sub_C63BB0(&v27, v22, v26, v21);
        v16 = v20;
        v17 = "";
        v23 = v19;
        v27 = (__int64 *)&v29;
        v18 = "i8 must be 8-bit aligned";
        goto LABEL_14;
      }
      v25 = 0;
      if ( (unsigned int)v33 <= 2 )
      {
LABEL_21:
        v6 = (unsigned int)v5;
        sub_AE24A0(a2, v5, v13, v24, v25);
        *a1 = 1;
        goto LABEL_5;
      }
      goto LABEL_16;
    }
  }
  v25 = v24;
  if ( (unsigned int)v33 > 2 )
  {
LABEL_16:
    v6 = v32[4];
    sub_AE1890(&v27, v6, v32[5], &v25, (__int64)"preferred", 9, 0);
    v13 = v22;
    v12 = (unsigned __int64)v27 & 0xFFFFFFFFFFFFFFFELL;
    if ( ((unsigned __int64)v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
LABEL_17:
      *a1 = v12 | 1;
      goto LABEL_5;
    }
  }
  if ( v25 >= v24 )
  {
    LODWORD(v13) = v26;
    goto LABEL_21;
  }
  v23 = sub_C63BB0(&v27, v6, v13, v14);
  v16 = v15;
  v17 = "";
  v27 = (__int64 *)&v29;
  v18 = "preferred alignment cannot be less than the ABI alignment";
LABEL_14:
  sub_AE11D0((__int64 *)&v27, v18, (__int64)v17);
  v6 = (__int64)&v27;
  sub_C63F00(a1, &v27, v23, v16);
  if ( v27 != (__int64 *)&v29 )
  {
    v6 = (__int64)(v29 + 1);
    j_j___libc_free_0(v27, v29 + 1);
  }
LABEL_5:
  if ( v32 != (__int64 *)v34 )
    _libc_free(v32, v6);
  return a1;
}
