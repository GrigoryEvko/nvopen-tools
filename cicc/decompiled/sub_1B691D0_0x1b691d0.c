// Function: sub_1B691D0
// Address: 0x1b691d0
//
__int64 __fastcall sub_1B691D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v4; // rdx
  char *v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rdx
  char *v11; // rax
  __int64 v12; // rdx
  const char *v13; // rax
  __int64 v14; // rdx
  __int64 v16; // rdx
  unsigned __int8 v17; // [rsp+Fh] [rbp-1D1h]
  __int64 v18; // [rsp+18h] [rbp-1C8h]
  _QWORD *s2; // [rsp+20h] [rbp-1C0h]
  __int64 n; // [rsp+30h] [rbp-1B0h]
  _QWORD v22[2]; // [rsp+40h] [rbp-1A0h] BYREF
  _QWORD v23[2]; // [rsp+50h] [rbp-190h] BYREF
  __int64 v24; // [rsp+60h] [rbp-180h]
  char *v25; // [rsp+70h] [rbp-170h]
  char v26; // [rsp+80h] [rbp-160h]
  char v27; // [rsp+81h] [rbp-15Fh]
  _QWORD v28[2]; // [rsp+90h] [rbp-150h] BYREF
  __int64 v29; // [rsp+A0h] [rbp-140h]
  __int64 v30; // [rsp+B0h] [rbp-130h]
  __int16 v31; // [rsp+C0h] [rbp-120h]
  _QWORD v32[2]; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v33; // [rsp+E0h] [rbp-100h]
  __int128 v34; // [rsp+F0h] [rbp-F0h]
  __int64 v35; // [rsp+100h] [rbp-E0h]
  _QWORD v36[2]; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v37; // [rsp+120h] [rbp-C0h]
  __int128 v38; // [rsp+130h] [rbp-B0h]
  __int64 v39; // [rsp+140h] [rbp-A0h]
  _QWORD *v40; // [rsp+150h] [rbp-90h] BYREF
  __int64 v41; // [rsp+158h] [rbp-88h]
  _QWORD v42[2]; // [rsp+160h] [rbp-80h] BYREF
  _QWORD *v43; // [rsp+170h] [rbp-70h] BYREF
  __int64 v44; // [rsp+178h] [rbp-68h]
  _QWORD v45[2]; // [rsp+180h] [rbp-60h] BYREF
  _QWORD *v46; // [rsp+190h] [rbp-50h] BYREF
  _QWORD *v47; // [rsp+198h] [rbp-48h]
  _QWORD v48[8]; // [rsp+1A0h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a2 + 16);
  v18 = a2 + 8;
  if ( a2 + 8 != v2 )
  {
    v17 = 0;
    while ( 1 )
    {
      v8 = v2 - 56;
      v9 = *(_QWORD *)(a1 + 16);
      if ( !v2 )
        v8 = 0;
      v10 = *(_QWORD *)(a1 + 24);
      LOBYTE(v42[0]) = 0;
      v41 = 0;
      v40 = v42;
      sub_16C9340((__int64)&v46, v9, v10, 0);
      v11 = (char *)sub_1649960(v8);
      sub_16C97A0((__int64)&v43, (__int64 *)&v46, *(char **)(a1 + 48), *(_QWORD *)(a1 + 56), v11, v12, &v40);
      sub_16C93F0(&v46);
      if ( v41 )
      {
        LOWORD(v39) = 260;
        *(_QWORD *)&v38 = &v40;
        *(_QWORD *)&v34 = ": ";
        LOWORD(v35) = 259;
        v30 = a2 + 176;
        v31 = 260;
        v27 = 1;
        v25 = " in ";
        v26 = 3;
        v22[0] = sub_1649960(v8);
        v23[0] = "unable to transforn ";
        v23[1] = v22;
        v22[1] = v16;
        LOWORD(v24) = 1283;
        v28[1] = " in ";
        v28[0] = v23;
        LOWORD(v29) = 770;
        v32[0] = v28;
        v32[1] = a2 + 176;
        LOWORD(v33) = 1026;
        v36[0] = v32;
        v36[1] = ": ";
        LOWORD(v37) = 770;
        v46 = v36;
        v47 = &v40;
        LOWORD(v48[0]) = 1026;
        sub_16BCFB0((__int64)&v46, 1u);
      }
      n = v44;
      s2 = v43;
      v13 = sub_1649960(v8);
      if ( n != v14 || n && memcmp(v13, s2, n) )
        break;
      if ( v43 != v45 )
        j_j___libc_free_0(v43, v45[0] + 1LL);
      if ( v40 == v42 )
      {
LABEL_15:
        v2 = *(_QWORD *)(v2 + 8);
        if ( v18 == v2 )
          return v17;
      }
      else
      {
        j_j___libc_free_0(v40, v42[0] + 1LL);
        v2 = *(_QWORD *)(v2 + 8);
        if ( v18 == v2 )
          return v17;
      }
    }
    if ( v8 )
    {
      v5 = (char *)sub_1649960(v8);
      v46 = v48;
      if ( v5 )
      {
        sub_1B678F0((__int64 *)&v46, v5, (__int64)&v5[v4]);
      }
      else
      {
        v47 = 0;
        LOBYTE(v48[0]) = 0;
      }
      sub_1B679A0(a2, v8, (__int64)&v46, (__int64)&v43);
      if ( v46 != v48 )
        j_j___libc_free_0(v46, v48[0] + 1LL);
    }
    v6 = sub_16321C0(a2, (__int64)v43, v44, 0);
    if ( v6 )
    {
      v7 = sub_16498B0(v6);
      sub_164B0D0(v8, v7);
    }
    else
    {
      v46 = &v43;
      LOWORD(v48[0]) = 260;
      sub_164B780(v8, (__int64 *)&v46);
    }
    if ( v43 != v45 )
      j_j___libc_free_0(v43, v45[0] + 1LL);
    if ( v40 != v42 )
      j_j___libc_free_0(v40, v42[0] + 1LL);
    v17 = 1;
    goto LABEL_15;
  }
  return 0;
}
