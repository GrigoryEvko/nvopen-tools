// Function: sub_1CF4B50
// Address: 0x1cf4b50
//
__int64 __fastcall sub_1CF4B50(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  char v10; // r8
  __int64 result; // rax
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  double v27; // xmm4_8
  double v28; // xmm5_8
  _QWORD *v29; // rbx
  _QWORD *v30; // r12
  __int64 v31; // rdi
  _QWORD *v32; // rbx
  _QWORD *v33; // rdi
  _QWORD v34[4]; // [rsp+0h] [rbp-A0h] BYREF
  char v35; // [rsp+20h] [rbp-80h]
  _QWORD *v36; // [rsp+28h] [rbp-78h]
  _QWORD v37[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v38; // [rsp+40h] [rbp-60h]
  __int64 v39; // [rsp+48h] [rbp-58h]
  int v40; // [rsp+50h] [rbp-50h]
  __int64 v41; // [rsp+58h] [rbp-48h]
  __int64 v42; // [rsp+60h] [rbp-40h]
  __int64 v43; // [rsp+68h] [rbp-38h]
  int v44; // [rsp+70h] [rbp-30h]
  int v45; // [rsp+78h] [rbp-28h]

  v10 = sub_1636880(a1, a2);
  result = 0;
  if ( !v10 )
  {
    v12 = *(__int64 **)(a1 + 8);
    v13 = *v12;
    v14 = v12[1];
    if ( v13 == v14 )
      goto LABEL_21;
    while ( *(_UNKNOWN **)v13 != &unk_4F9E06C )
    {
      v13 += 16;
      if ( v14 == v13 )
        goto LABEL_21;
    }
    v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(
            *(_QWORD *)(v13 + 8),
            &unk_4F9E06C);
    v16 = *(__int64 **)(a1 + 8);
    v17 = v15 + 160;
    v18 = *v16;
    v19 = v16[1];
    if ( v18 == v19 )
      goto LABEL_21;
    while ( *(_UNKNOWN **)v18 != &unk_4F9920C )
    {
      v18 += 16;
      if ( v19 == v18 )
        goto LABEL_21;
    }
    v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(
            *(_QWORD *)(v18 + 8),
            &unk_4F9920C);
    v21 = *(__int64 **)(a1 + 8);
    v22 = v20 + 160;
    v23 = *v21;
    v24 = v21[1];
    if ( v23 == v24 )
LABEL_21:
      BUG();
    while ( *(_UNKNOWN **)v23 != &unk_505440C )
    {
      v23 += 16;
      if ( v24 == v23 )
        goto LABEL_21;
    }
    v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(
            *(_QWORD *)(v23 + 8),
            &unk_505440C);
    v34[0] = a2;
    v26 = *(_QWORD *)(v25 + 160);
    LOBYTE(v25) = *(_BYTE *)(a1 + 153);
    v34[1] = v17;
    v34[3] = v22;
    v34[2] = v26;
    v35 = v25;
    v36 = 0;
    v37[0] = 0;
    v37[1] = 0;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    v45 = 0;
    sub_1CF4130(v34, (__int64)&unk_505440C, v26, a3, a4, a5, a6, v27, v28, a9, a10);
    sub_1CF1600(v37, a1 + 160);
    j___libc_free_0(v42);
    j___libc_free_0(v38);
    v29 = (_QWORD *)v37[0];
    if ( v37[0] )
    {
      do
      {
        v30 = v29;
        v29 = (_QWORD *)*v29;
        v31 = v30[1];
        if ( v31 )
          j_j___libc_free_0(v31, v30[3] - v31);
        j_j___libc_free_0(v30, 32);
      }
      while ( v29 );
    }
    v32 = v36;
    while ( v32 )
    {
      v33 = v32;
      v32 = (_QWORD *)*v32;
      j_j___libc_free_0(v33, 16);
    }
    return 1;
  }
  return result;
}
