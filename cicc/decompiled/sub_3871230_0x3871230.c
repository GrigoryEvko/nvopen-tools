// Function: sub_3871230
// Address: 0x3871230
//
__int64 __fastcall sub_3871230(__int64 *a1, __int64 *a2, __int64 a3, _QWORD *a4, __m128i a5, __m128i a6)
{
  unsigned int v9; // eax
  __int64 v10; // r15
  unsigned int v11; // r14d
  __int16 v12; // ax
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 v17; // rcx
  int v18; // eax
  bool v19; // r10
  __int64 v20; // rax
  unsigned int v21; // eax
  __int64 v22; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  bool v26; // cc
  _BYTE *v27; // rsi
  __int64 v28; // rax
  __int64 *v29; // r12
  __int64 v30; // rax
  bool v31; // r10
  __int64 v32; // rax
  __int64 *v33; // rdi
  unsigned int v34; // r10d
  int v35; // [rsp+4h] [rbp-9Ch]
  int v36; // [rsp+4h] [rbp-9Ch]
  bool v37; // [rsp+4h] [rbp-9Ch]
  __int64 v38; // [rsp+8h] [rbp-98h]
  __int64 v39; // [rsp+8h] [rbp-98h]
  bool v40; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+18h] [rbp-88h]
  bool v43; // [rsp+18h] [rbp-88h]
  __int64 v44; // [rsp+28h] [rbp-78h] BYREF
  __int64 v45; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v46; // [rsp+38h] [rbp-68h]
  __int64 *v47; // [rsp+40h] [rbp-60h] BYREF
  __int64 v48; // [rsp+48h] [rbp-58h]
  _BYTE v49[80]; // [rsp+50h] [rbp-50h] BYREF

  LOBYTE(v9) = sub_1456110(a3);
  if ( (_BYTE)v9 )
    return 1;
  v10 = *a1;
  if ( a3 == *a1 )
  {
    v22 = sub_1456040(a3);
    *a1 = sub_145CF80((__int64)a4, v22, 1, 0);
    return 1;
  }
  v11 = v9;
  v12 = *(_WORD *)(v10 + 24);
  if ( v12 )
    goto LABEL_12;
  if ( sub_14560B0(*a1) )
    return 1;
  if ( !*(_WORD *)(a3 + 24) )
  {
    sub_16A9F90((__int64)&v47, *(_QWORD *)(v10 + 32) + 24LL, *(_QWORD *)(a3 + 32) + 24LL);
    v13 = (__int64 *)sub_15E0530(a4[3]);
    v14 = sub_159C0E0(v13, (__int64)&v47);
    v15 = v14;
    if ( (unsigned int)v48 > 0x40 && v47 )
    {
      v38 = v14;
      j_j___libc_free_0_0((unsigned __int64)v47);
      v15 = v38;
    }
    if ( *(_DWORD *)(v15 + 32) <= 0x40u )
    {
      if ( !*(_QWORD *)(v15 + 24) )
        goto LABEL_11;
    }
    else
    {
      v35 = *(_DWORD *)(v15 + 32);
      if ( v35 == (unsigned int)sub_16A57B0(v15 + 24) )
        goto LABEL_11;
    }
    *a1 = sub_145CE20((__int64)a4, v15);
    sub_16AB4D0((__int64)&v47, *(_QWORD *)(v10 + 32) + 24LL, *(_QWORD *)(a3 + 32) + 24LL);
    v24 = sub_145CF40((__int64)a4, (__int64)&v47);
    v25 = sub_13A5B00((__int64)a4, *a2, v24, 0, 0);
    v26 = (unsigned int)v48 <= 0x40;
    *a2 = v25;
    if ( !v26 && v47 )
      j_j___libc_free_0_0((unsigned __int64)v47);
    return 1;
  }
LABEL_11:
  v10 = *a1;
  v12 = *(_WORD *)(*a1 + 24);
LABEL_12:
  if ( v12 != 5 )
  {
LABEL_19:
    if ( v12 == 7 )
    {
      v44 = sub_13A5BC0((_QWORD *)v10, (__int64)a4);
      v20 = sub_1456040(v44);
      v45 = sub_145CF80((__int64)a4, v20, 0, 0);
      if ( (unsigned __int8)sub_3871230(&v44, &v45, a3, a4) )
      {
        if ( sub_14560B0(v45) )
        {
          v47 = **(__int64 ***)(v10 + 32);
          v21 = sub_3871230(&v47, a2, a3, a4);
          if ( (_BYTE)v21 )
          {
            v11 = v21;
            *a1 = sub_14799E0((__int64)a4, (__int64)v47, v44, *(_QWORD *)(v10 + 48), *(_WORD *)(v10 + 26) & 1);
          }
        }
      }
    }
    return v11;
  }
  v16 = *(__int64 **)(v10 + 32);
  if ( *(_WORD *)(*v16 + 24) )
    return v11;
  v39 = *v16;
  sub_16AB4D0((__int64)&v47, *(_QWORD *)(*v16 + 32) + 24LL, *(_QWORD *)(a3 + 32) + 24LL);
  v17 = v39;
  if ( (unsigned int)v48 <= 0x40 )
  {
    v19 = v47 == 0;
  }
  else
  {
    v36 = v48;
    v18 = sub_16A57B0((__int64)&v47);
    v17 = v39;
    v19 = v36 == v18;
    if ( v47 )
    {
      v37 = v36 == v18;
      j_j___libc_free_0_0((unsigned __int64)v47);
      v19 = v37;
      v17 = v39;
    }
  }
  if ( !v19 )
  {
    v10 = *a1;
    v12 = *(_WORD *)(*a1 + 24);
    goto LABEL_19;
  }
  v27 = *(_BYTE **)(v10 + 32);
  v28 = *(_QWORD *)(v10 + 40);
  v40 = v19;
  v42 = v17;
  v47 = (__int64 *)v49;
  v48 = 0x400000000LL;
  sub_145C5B0((__int64)&v47, v27, &v27[8 * v28]);
  sub_16A9F90((__int64)&v45, *(_QWORD *)(v42 + 32) + 24LL, *(_QWORD *)(a3 + 32) + 24LL);
  v29 = v47;
  v30 = sub_145CF40((__int64)a4, (__int64)&v45);
  v31 = v40;
  *v29 = v30;
  if ( v46 > 0x40 && v45 )
  {
    j_j___libc_free_0_0(v45);
    v31 = v40;
  }
  v43 = v31;
  v32 = sub_147EE30(a4, &v47, 0, 0, a5, a6);
  v33 = v47;
  v34 = v43;
  *a1 = v32;
  if ( v33 != (__int64 *)v49 )
  {
    _libc_free((unsigned __int64)v33);
    return v43;
  }
  return v34;
}
