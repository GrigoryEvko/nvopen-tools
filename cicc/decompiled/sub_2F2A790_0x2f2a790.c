// Function: sub_2F2A790
// Address: 0x2f2a790
//
unsigned __int64 __fastcall sub_2F2A790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  unsigned int v6; // r15d
  unsigned int v7; // r13d
  __int64 v10; // rcx
  int v11; // edx
  int v12; // esi
  unsigned int i; // eax
  _DWORD *v14; // r8
  unsigned int v15; // eax
  int v16; // edx
  __int64 v17; // rdx
  __int64 v18; // rcx
  _BYTE *v19; // rdi
  unsigned __int64 v20; // r13
  __int64 v22; // r9
  char v23; // r8
  unsigned int v24; // r12d
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // r8
  __int64 v29; // r14
  __int32 v30; // eax
  __int64 v31; // rbx
  __int32 v32; // r12d
  __int64 v33; // rax
  __int64 v34; // r14
  unsigned __int8 *v35; // rsi
  _QWORD *v36; // r14
  __int64 v37; // rdx
  __int64 v38; // r12
  __int32 *v39; // r15
  unsigned int v40; // ebx
  __int32 v41; // edx
  __int32 v42; // ecx
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // [rsp+18h] [rbp-E8h]
  __int64 v46; // [rsp+18h] [rbp-E8h]
  __int64 v47; // [rsp+18h] [rbp-E8h]
  __int64 v48; // [rsp+20h] [rbp-E0h]
  __int32 *v49; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v51; // [rsp+38h] [rbp-C8h] BYREF
  _BYTE *v52; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+48h] [rbp-B8h]
  _BYTE v54[16]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v55; // [rsp+60h] [rbp-A0h]
  __m128i v56; // [rsp+70h] [rbp-90h] BYREF
  __int64 v57; // [rsp+80h] [rbp-80h]
  __int64 v58; // [rsp+88h] [rbp-78h]
  __int64 v59; // [rsp+90h] [rbp-70h]
  __int32 *v60; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v61; // [rsp+A8h] [rbp-58h]
  _BYTE v62[80]; // [rsp+B0h] [rbp-50h] BYREF

  v6 = HIDWORD(a3);
  v7 = a3;
  v48 = a4 + 16;
  while ( 1 )
  {
    if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
    {
      v10 = v48;
      v11 = 3;
    }
    else
    {
      v16 = *(_DWORD *)(a4 + 24);
      v10 = *(_QWORD *)(a4 + 16);
      if ( !v16 )
      {
LABEL_46:
        v19 = v54;
        v18 = v6;
        goto LABEL_13;
      }
      v11 = v16 - 1;
    }
    v12 = 1;
    for ( i = v11
            & (((0xBF58476D1CE4E5B9LL * ((37 * v6) | ((unsigned __int64)(37 * v7) << 32))) >> 31)
             ^ (756364221 * v6)); ; i = v11 & v15 )
    {
      v14 = (_DWORD *)(v10 + 48LL * i);
      if ( *v14 == v7 && v14[1] == v6 )
        break;
      if ( *v14 == -1 && v14[1] == -1 )
        goto LABEL_46;
      v15 = v12 + i;
      ++v12;
    }
    v17 = (unsigned int)v14[4];
    v53 = 0x200000000LL;
    v52 = v54;
    if ( !(_DWORD)v17 )
    {
      v18 = v6;
      v19 = v54;
      goto LABEL_13;
    }
    v45 = v10 + 48LL * i;
    sub_2F29420((__int64)&v52, (__int64)(v14 + 2), v17, v10, (__int64)v14, 0xBF58476D1CE4E5B9LL);
    v22 = 0xBF58476D1CE4E5B9LL;
    v19 = v52;
    v55 = *(_QWORD *)(v45 + 40);
    if ( (int)v53 <= 0 )
    {
      v18 = v6;
LABEL_13:
      v20 = (v18 << 32) | v7;
LABEL_14:
      if ( v19 != v54 )
        _libc_free((unsigned __int64)v19);
      return v20;
    }
    if ( (_DWORD)v53 != 1 )
      break;
    v7 = *(_DWORD *)v52;
    v6 = *((_DWORD *)v52 + 1);
    if ( v52 != v54 )
      _libc_free((unsigned __int64)v52);
  }
  v23 = a5;
  v24 = v53;
  if ( v23 )
  {
    v25 = 0;
    v60 = (__int32 *)v62;
    v61 = 0x400000000LL;
    while ( 1 )
    {
      v26 = sub_2F2A790(a1, a2, *(_QWORD *)&v19[8 * v25], a4, 1, v22);
      v27 = (unsigned int)v61;
      v28 = (unsigned int)v61 + 1LL;
      if ( v28 > HIDWORD(v61) )
      {
        v47 = v26;
        sub_C8D5F0((__int64)&v60, v62, (unsigned int)v61 + 1LL, 8u, v28, v22);
        v27 = (unsigned int)v61;
        v26 = v47;
      }
      ++v25;
      *(_QWORD *)&v60[2 * v27] = v26;
      LODWORD(v61) = v61 + 1;
      if ( v24 <= (unsigned int)v25 )
        break;
      v19 = v52;
    }
    v29 = v55;
    v46 = v55;
    v30 = sub_2EC06C0(
            a1,
            *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (*v60 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
            byte_3F871B3,
            0,
            v28,
            v22);
    v31 = *(_QWORD *)(a2 + 8);
    v32 = v30;
    v33 = v29;
    v34 = *(_QWORD *)(v29 + 24);
    v35 = *(unsigned __int8 **)(v33 + 56);
    v51 = v35;
    if ( v35 )
    {
      sub_B96E90((__int64)&v51, (__int64)v35, 1);
      v56.m128i_i64[0] = (__int64)v51;
      if ( v51 )
      {
        sub_B976B0((__int64)&v51, v51, (__int64)&v56);
        v51 = 0;
      }
    }
    else
    {
      v56.m128i_i64[0] = 0;
    }
    v57 = 0;
    v56.m128i_i64[1] = 0;
    v36 = sub_2F2A600(v34, v46, v56.m128i_i64, v31, v32);
    v38 = v37;
    if ( v56.m128i_i64[0] )
      sub_B91220((__int64)&v56, v56.m128i_i64[0]);
    if ( v51 )
      sub_B91220((__int64)&v51, (__int64)v51);
    v49 = &v60[2 * (unsigned int)v61];
    if ( v60 != v49 )
    {
      v39 = v60;
      v40 = 2;
      do
      {
        v41 = v39[1];
        v42 = *v39;
        v39 += 2;
        v56.m128i_i32[2] = v42;
        v57 = 0;
        v56.m128i_i64[0] = (unsigned __int16)(v41 & 0xFFF) << 8;
        v58 = 0;
        v59 = 0;
        sub_2E8EAD0(v38, (__int64)v36, &v56);
        v43 = 5LL * v40;
        v40 += 2;
        v44 = *(_QWORD *)(*(_QWORD *)(v46 + 32) + 8 * v43 + 24);
        v56.m128i_i8[0] = 4;
        v57 = 0;
        v58 = v44;
        v56.m128i_i32[0] &= 0xFFF000FF;
        sub_2E8EAD0(v38, (__int64)v36, &v56);
        sub_2EBF120(a1, *(v39 - 2));
      }
      while ( v49 != v39 );
      v49 = v60;
    }
    v20 = ((unsigned __int64)((**(_DWORD **)(v38 + 32) >> 8) & 0xFFF) << 32)
        | *(unsigned int *)(*(_QWORD *)(v38 + 32) + 8LL);
    if ( v49 != (__int32 *)v62 )
      _libc_free((unsigned __int64)v49);
    v19 = v52;
    goto LABEL_14;
  }
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  return 0;
}
