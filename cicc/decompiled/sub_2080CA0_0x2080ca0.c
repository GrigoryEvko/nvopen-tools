// Function: sub_2080CA0
// Address: 0x2080ca0
//
bool __fastcall sub_2080CA0(
        __int64 *a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        __m128i a7,
        __m128i a8)
{
  __int64 v8; // r15
  bool result; // al
  __int64 *v11; // rbx
  __int64 v13; // r10
  __int64 v14; // r12
  _BYTE *v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // r8
  __int64 v18; // rcx
  unsigned __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdi
  unsigned int v22; // r13d
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rsi
  int v26; // edx
  int v27; // edx
  __int64 *v28; // rax
  unsigned int v29; // edx
  __int64 v30; // rax
  unsigned __int8 v31; // bl
  __int64 v32; // r12
  unsigned int v33; // eax
  __int64 v34; // r14
  const void **v35; // rdx
  const void **v36; // r12
  unsigned int v37; // ebx
  __int64 v38; // rsi
  __int64 v39; // r8
  __int64 v40; // r9
  int v41; // edx
  __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-C0h]
  __int64 v44; // [rsp+0h] [rbp-C0h]
  __int64 v45; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v46; // [rsp+8h] [rbp-B8h]
  __int64 *v47; // [rsp+10h] [rbp-B0h]
  char v48; // [rsp+10h] [rbp-B0h]
  __int64 *v49; // [rsp+18h] [rbp-A8h]
  __int64 v50; // [rsp+20h] [rbp-A0h]
  __int64 v51; // [rsp+20h] [rbp-A0h]
  unsigned int v52; // [rsp+28h] [rbp-98h]
  __int64 *v53; // [rsp+28h] [rbp-98h]
  _QWORD *v55; // [rsp+30h] [rbp-90h]
  __int64 *v56; // [rsp+38h] [rbp-88h]
  __int64 v57; // [rsp+80h] [rbp-40h] BYREF
  __int64 v58; // [rsp+88h] [rbp-38h]

  v8 = *a1;
  if ( *(_BYTE *)(*a1 + 16) != 56 )
    return 0;
  v11 = a1;
  v13 = a4;
  v14 = a5;
  v56 = *(__int64 **)(a5 + 552);
  v55 = (_QWORD *)v56[6];
  v15 = *(_BYTE **)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 16 )
  {
    v42 = sub_14C49D0(v15);
    v13 = a4;
    *v11 = v42;
    if ( !v42 )
      return 0;
  }
  else
  {
    *v11 = (__int64)v15;
  }
  v16 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
  v17 = v16 - 1;
  v18 = v16;
  if ( (unsigned int)v17 > 1 )
  {
    v52 = v16 - 1;
    v50 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
    v47 = a3;
    v45 = v13;
    v49 = v11;
    v19 = v8 + 24 * (v16 - 3 - (unsigned __int64)v16) + 24;
    v43 = v14;
    v20 = v8 - 24LL * v16;
    while ( 1 )
    {
      v21 = *(_QWORD *)(v20 + 24);
      if ( *(_BYTE *)(v21 + 16) != 13 )
        return 0;
      v22 = *(_DWORD *)(v21 + 32);
      if ( v22 <= 0x40 )
      {
        if ( *(_QWORD *)(v21 + 24) )
          return 0;
      }
      else if ( v22 != (unsigned int)sub_16A57B0(v21 + 24) )
      {
        return 0;
      }
      v20 += 24;
      if ( v19 == v20 )
      {
        v17 = v52;
        v18 = v50;
        v11 = v49;
        a3 = v47;
        v13 = v45;
        v14 = v43;
        break;
      }
    }
  }
  v51 = v13;
  v53 = *(__int64 **)(v8 + 24 * (v17 - v18));
  if ( !sub_2052A80(v14, *v11) )
    return 0;
  v48 = sub_2052A80(v14, (__int64)v53);
  if ( !v48 )
    return 0;
  v44 = sub_1E0A0C0(v56[4]);
  v46 = sub_2046180(v44, 0);
  sub_204D410((__int64)&v57, *(_QWORD *)v14, *(_DWORD *)(v14 + 536));
  v23 = sub_12BE0A0(v44, *(_QWORD *)(v8 + 64));
  v24 = sub_1D38BB0((__int64)v56, v23, (__int64)&v57, v46, 0, 1, a6, *(double *)a7.m128i_i64, a8, 0);
  v25 = v57;
  *(_QWORD *)v51 = v24;
  *(_DWORD *)(v51 + 8) = v26;
  if ( v25 )
    sub_161E7C0((__int64)&v57, v25);
  *(_QWORD *)a2 = sub_20685E0(v14, (__int64 *)*v11, a6, a7, a8);
  *(_DWORD *)(a2 + 8) = v27;
  v28 = sub_20685E0(v14, v53, a6, a7, a8);
  *a3 = (__int64)v28;
  *((_DWORD *)a3 + 2) = v29;
  v30 = v28[5] + 16LL * v29;
  v31 = *(_BYTE *)v30;
  v32 = *(_QWORD *)(v30 + 8);
  LOBYTE(v57) = v31;
  v58 = v32;
  if ( v31 )
    result = (unsigned __int8)(v31 - 14) <= 0x5Fu;
  else
    result = sub_1F58D20((__int64)&v57);
  if ( !result )
  {
    v33 = sub_1F7DEB0(v55, v31, v32, *(_QWORD *)(*(_QWORD *)v8 + 32LL), 0);
    v34 = *a3;
    v36 = v35;
    v37 = v33;
    v38 = *(_QWORD *)(*a3 + 72);
    v57 = v38;
    if ( v38 )
      sub_1623A60((__int64)&v57, v38, 2);
    v39 = *a3;
    v40 = a3[1];
    LODWORD(v58) = *(_DWORD *)(v34 + 64);
    *a3 = (__int64)sub_1D35F20(
                     v56,
                     v37,
                     v36,
                     (__int64)&v57,
                     v39,
                     v40,
                     *(double *)a6.m128i_i64,
                     *(double *)a7.m128i_i64,
                     a8);
    *((_DWORD *)a3 + 2) = v41;
    sub_17CD270(&v57);
    return v48;
  }
  return result;
}
