// Function: sub_AE21F0
// Address: 0xae21f0
//
unsigned __int64 *__fastcall sub_AE21F0(unsigned __int64 *a1, __int64 a2, const char *a3, __int64 a4)
{
  const char **v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdx
  const char **v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rcx
  unsigned __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // r15
  char *v26; // rdx
  char *v27; // rsi
  __int64 v28; // rdx
  unsigned __int8 v29; // dl
  __int64 v30; // [rsp-10h] [rbp-D0h]
  const char **v31; // [rsp-8h] [rbp-C8h]
  unsigned int v32; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v33; // [rsp+1Eh] [rbp-A2h] BYREF
  unsigned __int8 v34; // [rsp+1Fh] [rbp-A1h] BYREF
  const char *v35; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+28h] [rbp-98h]
  __int64 v37; // [rsp+30h] [rbp-90h] BYREF
  __int64 v38; // [rsp+38h] [rbp-88h]
  __int64 v39; // [rsp+40h] [rbp-80h]
  _QWORD *v40; // [rsp+50h] [rbp-70h] BYREF
  __int64 v41; // [rsp+58h] [rbp-68h]
  _BYTE v42[96]; // [rsp+60h] [rbp-60h] BYREF

  v40 = v42;
  v41 = 0x300000000LL;
  if ( a4 )
  {
    --a4;
    ++a3;
  }
  v35 = a3;
  v6 = (const char **)&v40;
  v36 = a4;
  sub_C93960(&v35, &v40, 58, 0xFFFFFFFFLL, 1);
  if ( (unsigned __int64)(unsigned int)v41 - 2 > 1 )
  {
    LOWORD(v39) = 259;
    v35 = "a:<abi>[:<pref>]";
    sub_AE1520((__int64)a1, (__int64)&v40, v7, v8, v9, v10, (__int64 *)"a:<abi>[:<pref>]", v36, v37, v38, 259);
    goto LABEL_5;
  }
  v12 = v40;
  v13 = v40[1];
  if ( v13 )
  {
    v14 = *v40;
    if ( (unsigned __int8)sub_C93C90(*v40, v13, 10, &v35)
      || (v15 = (unsigned int)v35, v35 != (const char *)(unsigned int)v35)
      || (_DWORD)v35 )
    {
      v32 = sub_C63BB0(v14, v13, v15, v16, v17);
      v25 = v28;
      v26 = "";
      v35 = (const char *)&v37;
      v27 = "size must be zero";
      goto LABEL_20;
    }
    v12 = v40;
  }
  v18 = v12[3];
  v33 = 0;
  sub_AE1890(&v35, v12[2], v18, &v33, (__int64)"ABI", 3, 1u);
  v21 = v30;
  v6 = v31;
  v22 = (unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL;
  if ( ((unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL) != 0
    || (v34 = v33, (unsigned int)v41 > 2)
    && (v6 = (const char **)v40[4],
        sub_AE1890(&v35, (__int64)v6, v40[5], &v34, (__int64)"preferred", 9, 0),
        v19 = v31,
        v22 = (unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL,
        ((unsigned __int64)v35 & 0xFFFFFFFFFFFFFFFELL) != 0) )
  {
    *a1 = v22 | 1;
    goto LABEL_5;
  }
  if ( v34 >= v33 )
  {
    v29 = v33;
    *(_BYTE *)(a2 + 481) = v34;
    *(_BYTE *)(a2 + 480) = v29;
    *a1 = 1;
    goto LABEL_5;
  }
  v23 = sub_C63BB0(&v35, v6, v19, v21, v20);
  v25 = v24;
  v26 = "";
  v32 = v23;
  v35 = (const char *)&v37;
  v27 = "preferred alignment cannot be less than the ABI alignment";
LABEL_20:
  sub_AE11D0((__int64 *)&v35, v27, (__int64)v26);
  v6 = &v35;
  sub_C63F00(a1, &v35, v32, v25);
  if ( v35 != (const char *)&v37 )
  {
    v6 = (const char **)(v37 + 1);
    j_j___libc_free_0(v35, v37 + 1);
  }
LABEL_5:
  if ( v40 != (_QWORD *)v42 )
    _libc_free(v40, v6);
  return a1;
}
