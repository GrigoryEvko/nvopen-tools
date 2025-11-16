// Function: sub_D87E70
// Address: 0xd87e70
//
__int64 __fastcall sub_D87E70(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  _BYTE *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  const void *v13; // r14
  __int64 v14; // rax
  size_t v15; // rdx
  unsigned __int64 v16; // r15
  _QWORD *v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rcx
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  _QWORD *v24; // rsi
  _QWORD *v25; // rdx
  bool v26; // cc
  __int32 v27; // eax
  __int64 v28; // rdi
  _QWORD *v30; // [rsp+28h] [rbp-F8h]
  size_t v31; // [rsp+30h] [rbp-F0h]
  __int64 v32[2]; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v33[2]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v34; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v35; // [rsp+68h] [rbp-B8h]
  __int64 v36; // [rsp+70h] [rbp-B0h]
  unsigned int v37; // [rsp+78h] [rbp-A8h]
  __int64 v38; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v39; // [rsp+88h] [rbp-98h]
  __int64 v40; // [rsp+90h] [rbp-90h]
  unsigned int v41; // [rsp+98h] [rbp-88h]
  __m128i v42; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v43; // [rsp+B0h] [rbp-70h] BYREF
  int v44; // [rsp+B8h] [rbp-68h]
  int v45; // [rsp+C8h] [rbp-58h] BYREF
  __int64 v46; // [rsp+D0h] [rbp-50h]
  int *v47; // [rsp+D8h] [rbp-48h]
  int *v48; // [rsp+E0h] [rbp-40h]
  __int64 v49; // [rsp+E8h] [rbp-38h]

  sub_AADB10((__int64)v32, *(_DWORD *)(a1 + 8), 1);
  v4 = *(_QWORD *)(a1 + 96);
  v45 = 0;
  v46 = 0;
  v47 = &v45;
  v48 = &v45;
  v49 = 0;
  if ( !v4 )
    goto LABEL_57;
  v5 = *(_DWORD *)(a1 + 88);
  v46 = v4;
  v45 = v5;
  v47 = *(int **)(a1 + 104);
  v48 = *(int **)(a1 + 112);
  *(_QWORD *)(v4 + 8) = &v45;
  v6 = *(_QWORD *)(a1 + 120);
  v7 = (__int64)v47;
  *(_QWORD *)(a1 + 96) = 0;
  v49 = v6;
  *(_QWORD *)(a1 + 104) = a1 + 88;
  *(_QWORD *)(a1 + 112) = a1 + 88;
  *(_QWORD *)(a1 + 120) = 0;
  if ( (int *)v7 == &v45 )
  {
LABEL_56:
    v4 = v46;
LABEL_57:
    sub_D85C20(v4);
    sub_969240(v33);
    return sub_969240(v32);
  }
  v30 = (_QWORD *)(a2 + 8);
  while ( 1 )
  {
    v9 = *(_BYTE **)(v7 + 32);
    if ( v9 )
    {
      while ( !sub_B2FC80((__int64)v9) && !(unsigned __int8)sub_B2F6B0((__int64)v9) && (v9[33] & 0x40) != 0 )
      {
        if ( !*v9 )
        {
          v8 = *(_QWORD *)(v7 + 40);
          v42.m128i_i64[0] = (__int64)v9;
          v42.m128i_i64[1] = v8;
          sub_D87C70((_QWORD *)(a1 + 80), &v42, v7 + 48);
          goto LABEL_5;
        }
        if ( *v9 == 1 )
        {
          v10 = sub_B325F0((__int64)v9);
          if ( v9 != (_BYTE *)v10 )
          {
            v9 = (_BYTE *)v10;
            if ( v10 )
              continue;
          }
        }
        break;
      }
    }
    if ( !a2 )
      break;
    v11 = *(_QWORD *)(v7 + 32);
    v12 = *(_QWORD *)(v11 + 40);
    v13 = *(const void **)(v12 + 168);
    v31 = *(_QWORD *)(v12 + 176);
    sub_B2F930(&v42, v11);
    v14 = sub_B2F650(v42.m128i_i64[0], v42.m128i_i64[1]);
    v15 = v31;
    v16 = v14;
    if ( (__int64 *)v42.m128i_i64[0] != &v43 )
    {
      j_j___libc_free_0(v42.m128i_i64[0], v43 + 1);
      v15 = v31;
    }
    v17 = *(_QWORD **)(a2 + 16);
    if ( v17 )
    {
      v18 = v30;
      do
      {
        while ( 1 )
        {
          v19 = v17[2];
          v20 = v17[3];
          if ( v16 <= v17[4] )
            break;
          v17 = (_QWORD *)v17[3];
          if ( !v20 )
            goto LABEL_22;
        }
        v18 = v17;
        v17 = (_QWORD *)v17[2];
      }
      while ( v19 );
LABEL_22:
      v21 = 0;
      if ( v30 != v18 && v16 >= v18[4] )
        v21 = (unsigned __int64)(v18 + 4) & 0xFFFFFFFFFFFFFFF8LL;
    }
    else
    {
      v21 = 0;
    }
    v22 = sub_D854A0(v21 | *(unsigned __int8 *)(a2 + 343), v13, v15);
    if ( !v22 )
      break;
    v23 = *(_QWORD *)(v22 + 88);
    if ( !v23 )
      break;
    v24 = *(_QWORD **)v23;
    v25 = *(_QWORD **)(v23 + 8);
    if ( v24 == v25 )
      break;
    while ( *v24 != (unsigned int)*(_QWORD *)(v7 + 40) )
    {
      v24 += 8;
      if ( v25 == v24 )
        goto LABEL_55;
    }
    if ( sub_AAF760((__int64)(v24 + 1)) )
      break;
    sub_AB4E00((__int64)&v34, (__int64)(v24 + 1), *(_DWORD *)(a1 + 8));
    if ( !sub_AAF7D0((__int64)&v34) )
    {
      sub_D87200((__int64)&v38, (__int64)&v34, v7 + 48);
      sub_D87290((__int64)&v42, a1, (__int64)&v38);
      if ( *(_DWORD *)(a1 + 8) > 0x40u && *(_QWORD *)a1 )
        j_j___libc_free_0_0(*(_QWORD *)a1);
      v26 = *(_DWORD *)(a1 + 24) <= 0x40u;
      *(_QWORD *)a1 = v42.m128i_i64[0];
      v27 = v42.m128i_i32[2];
      v42.m128i_i32[2] = 0;
      *(_DWORD *)(a1 + 8) = v27;
      if ( v26 || (v28 = *(_QWORD *)(a1 + 16)) == 0 )
      {
        *(_QWORD *)(a1 + 16) = v43;
        *(_DWORD *)(a1 + 24) = v44;
      }
      else
      {
        j_j___libc_free_0_0(v28);
        v26 = v42.m128i_i32[2] <= 0x40u;
        *(_QWORD *)(a1 + 16) = v43;
        *(_DWORD *)(a1 + 24) = v44;
        if ( !v26 && v42.m128i_i64[0] )
          j_j___libc_free_0_0(v42.m128i_i64[0]);
      }
      if ( v41 > 0x40 && v40 )
        j_j___libc_free_0_0(v40);
      if ( v39 > 0x40 && v38 )
        j_j___libc_free_0_0(v38);
    }
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
    if ( v35 > 0x40 )
    {
      if ( v34 )
        j_j___libc_free_0_0(v34);
    }
LABEL_5:
    v7 = sub_220EEE0(v7);
    if ( (int *)v7 == &v45 )
      goto LABEL_56;
  }
LABEL_55:
  sub_D87370(a1, (__int64)v32);
  sub_D85C20(v46);
  sub_969240(v33);
  return sub_969240(v32);
}
