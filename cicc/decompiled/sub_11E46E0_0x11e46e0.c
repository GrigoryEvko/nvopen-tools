// Function: sub_11E46E0
// Address: 0x11e46e0
//
__int64 __fastcall sub_11E46E0(_QWORD *a1, __int64 a2, unsigned int **a3)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // r13
  _DWORD **v6; // rbx
  __int64 v7; // r15
  _DWORD *v8; // rax
  _DWORD *v9; // rax
  __int64 *v10; // r14
  char v11; // al
  __int64 v12; // r14
  _QWORD *v14; // r15
  _QWORD *v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  _BYTE *v18; // rax
  _QWORD *i; // r15
  unsigned __int8 *v20; // rax
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rax
  __m128i v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rax
  bool v28; // al
  _DWORD *v29; // [rsp+8h] [rbp-E8h]
  __int64 v30; // [rsp+10h] [rbp-E0h]
  __int64 v31; // [rsp+18h] [rbp-D8h]
  __int64 *v32; // [rsp+20h] [rbp-D0h]
  __int64 *v34; // [rsp+30h] [rbp-C0h]
  bool v35; // [rsp+3Fh] [rbp-B1h]
  _DWORD *v37; // [rsp+48h] [rbp-A8h]
  _DWORD *v38; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v39; // [rsp+58h] [rbp-98h]
  __m128i v40; // [rsp+70h] [rbp-80h] BYREF
  __int64 v41; // [rsp+80h] [rbp-70h]
  __int64 v42; // [rsp+88h] [rbp-68h]
  __int64 v43; // [rsp+90h] [rbp-60h]
  __int64 v44; // [rsp+98h] [rbp-58h]
  __int64 v45; // [rsp+A0h] [rbp-50h]
  __int64 v46; // [rsp+A8h] [rbp-48h]
  __int16 v47; // [rsp+B0h] [rbp-40h]

  v3 = a2;
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v34 = *(__int64 **)(a2 - 32 * v4);
  v5 = *(_QWORD *)(a2 + 32 * (1 - v4));
  v32 = (__int64 *)sub_B43CA0(a2);
  v6 = (_DWORD **)(v5 + 24);
  v30 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)v5 != 18 )
  {
    v17 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
    if ( (unsigned int)v17 > 1 )
      return 0;
    if ( *(_BYTE *)v5 > 0x15u )
      return 0;
    v18 = sub_AD7630(v5, 0, v17);
    if ( !v18 || *v18 != 18 )
      return 0;
    v6 = (_DWORD **)(v18 + 24);
  }
  v29 = sub_C33320();
  sub_C3B1B0((__int64)&v40, 0.5);
  sub_C407B0(&v38, v40.m128i_i64, v29);
  sub_C338F0((__int64)&v40);
  sub_C41640((__int64 *)&v38, *v6, 1, (bool *)v40.m128i_i8);
  v7 = (__int64)v38;
  v31 = (__int64)*v6;
  v8 = sub_C33340();
  v35 = 0;
  v37 = v8;
  if ( v31 == v7 )
  {
    if ( (_DWORD *)v7 != v8 )
    {
      v35 = sub_C33D00((__int64)v6, (__int64)&v38);
      if ( v38 != v37 )
        goto LABEL_4;
      goto LABEL_26;
    }
    v28 = sub_C3E590((__int64)v6, (__int64)&v38);
    v7 = (__int64)v38;
    v35 = v28;
  }
  if ( (_DWORD *)v7 != v37 )
  {
LABEL_4:
    sub_C338F0((__int64)&v38);
    goto LABEL_5;
  }
LABEL_26:
  if ( v39 )
  {
    v14 = &v39[3 * *(v39 - 1)];
    if ( v39 != v14 )
    {
      v15 = &v39[3 * *(v39 - 1)];
      do
      {
        v15 -= 3;
        sub_91D830(v15);
      }
      while ( v39 != v15 );
      v14 = v15;
      v3 = a2;
    }
    j_j_j___libc_free_0_0(v14 - 1);
    if ( v35 )
      goto LABEL_6;
    goto LABEL_32;
  }
LABEL_5:
  if ( v35 )
    goto LABEL_6;
LABEL_32:
  sub_C3B1B0((__int64)&v40, -0.5);
  sub_C407B0(&v38, v40.m128i_i64, v29);
  sub_C338F0((__int64)&v40);
  sub_C41640((__int64 *)&v38, *v6, 1, (bool *)v40.m128i_i8);
  v16 = (__int64)v38;
  if ( *v6 == v38 )
  {
    if ( v37 == v38 )
      v35 = sub_C3E590((__int64)v6, (__int64)&v38);
    else
      v35 = sub_C33D00((__int64)v6, (__int64)&v38);
    v16 = (__int64)v38;
  }
  if ( (_DWORD *)v16 == v37 )
  {
    if ( v39 )
    {
      for ( i = &v39[3 * *(v39 - 1)]; v39 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v38);
  }
  if ( !v35 )
    return 0;
LABEL_6:
  if ( v37 == *v6 )
    v9 = v6[1];
  else
    v9 = v6;
  if ( (v9[5] & 8) != 0 && !(unsigned __int8)sub_B45200(v3) && !sub_B451B0(v3) )
    return 0;
  if ( !sub_B49E00(v3) && !sub_B451D0(v3) )
  {
    v41 = 0;
    v44 = v3;
    v24.m128i_i64[1] = a1[3];
    v25 = a1[6];
    v46 = 0;
    v26 = a1[4];
    v27 = a1[5];
    v24.m128i_i64[0] = a1[2];
    v42 = v26;
    v40 = v24;
    v43 = v25;
    v47 = 257;
    v45 = v27;
    if ( (sub_9B4030(v34, 516, 0, &v40) & 0x204) != 0 )
      return 0;
  }
  v10 = (__int64 *)a1[3];
  v11 = sub_B49E00(v3);
  v12 = sub_11D9A40((__int64)v34, 0, v11, v32, (__int64)a3, v10);
  if ( !v12 )
    return 0;
  if ( sub_B451E0(v3)
    || (BYTE4(v38) = 0,
        v40.m128i_i64[0] = (__int64)"abs",
        LOWORD(v43) = 259,
        (v12 = sub_B33BC0((__int64)a3, 0xAAu, v12, (__int64)v38, (__int64)&v40)) != 0) )
  {
    if ( *(_BYTE *)v12 == 85 )
      *(_WORD *)(v12 + 2) = *(_WORD *)(v12 + 2) & 0xFFFC | *(_WORD *)(v3 + 2) & 3;
  }
  if ( !sub_B451D0(v3) )
  {
    v21 = sub_AD9500(v30, 0);
    v22 = sub_AD9500(v30, 1);
    LOWORD(v43) = 259;
    HIDWORD(v38) = 0;
    v40.m128i_i64[0] = (__int64)"isinf";
    v23 = sub_B35C90((__int64)a3, 1u, (__int64)v34, v22, (__int64)&v40, 0, (unsigned int)v38, 0);
    LOWORD(v43) = 257;
    v12 = sub_B36550(a3, v23, v21, v12, (__int64)&v40, 0);
  }
  if ( v37 == *v6 )
    v6 = (_DWORD **)v6[1];
  if ( (*((_BYTE *)v6 + 20) & 8) != 0 )
  {
    v40.m128i_i64[0] = (__int64)"reciprocal";
    LOWORD(v43) = 259;
    v20 = sub_AD8DD0(v30, 1.0);
    HIDWORD(v38) = 0;
    return sub_A82920(a3, v20, (_BYTE *)v12, (unsigned int)v38, (__int64)&v40, 0);
  }
  return v12;
}
