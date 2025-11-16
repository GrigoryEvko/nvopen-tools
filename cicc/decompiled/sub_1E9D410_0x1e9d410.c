// Function: sub_1E9D410
// Address: 0x1e9d410
//
unsigned __int64 __fastcall sub_1E9D410(size_t a1, __int64 a2, unsigned __int64 a3, __int64 a4, int a5)
{
  __int64 v6; // rcx
  unsigned int v7; // r14d
  __int64 v10; // rdi
  int v11; // esi
  int v12; // r10d
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // r9
  unsigned int i; // eax
  _DWORD *v16; // r9
  unsigned int v17; // eax
  int v18; // esi
  _BYTE *v19; // rdi
  unsigned __int64 v20; // r14
  __int64 v22; // rdx
  unsigned int v23; // r12d
  __int64 v24; // r14
  __int64 v25; // r8
  int v26; // r9d
  __int64 v27; // rax
  __int64 v28; // r15
  __int32 v29; // eax
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // r12
  __int32 *v33; // rbx
  __int32 v34; // edx
  __int32 v35; // ecx
  __int64 v36; // rdx
  int v37; // [rsp+18h] [rbp-D8h]
  __int64 v38; // [rsp+18h] [rbp-D8h]
  unsigned int v39; // [rsp+20h] [rbp-D0h]
  __int32 *v40; // [rsp+20h] [rbp-D0h]
  __int64 v41; // [rsp+20h] [rbp-D0h]
  __int64 v42; // [rsp+28h] [rbp-C8h]
  unsigned int v43; // [rsp+28h] [rbp-C8h]
  unsigned int v44; // [rsp+28h] [rbp-C8h]
  _BYTE *v45; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v46; // [rsp+38h] [rbp-B8h]
  _BYTE v47[16]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v48; // [rsp+50h] [rbp-A0h]
  __m128i v49; // [rsp+60h] [rbp-90h] BYREF
  __int64 v50; // [rsp+70h] [rbp-80h]
  __int64 v51; // [rsp+78h] [rbp-78h]
  __int64 v52; // [rsp+80h] [rbp-70h]
  __int32 *v53; // [rsp+90h] [rbp-60h] BYREF
  __int64 v54; // [rsp+98h] [rbp-58h]
  _BYTE v55[80]; // [rsp+A0h] [rbp-50h] BYREF

  v6 = HIDWORD(a3);
  v7 = a3;
  while ( 1 )
  {
    if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
    {
      v10 = a4 + 16;
      v11 = 3;
    }
    else
    {
      v18 = *(_DWORD *)(a4 + 24);
      v10 = *(_QWORD *)(a4 + 16);
      if ( !v18 )
      {
LABEL_11:
        v19 = v47;
        goto LABEL_12;
      }
      v11 = v18 - 1;
    }
    v12 = 1;
    v13 = ((((unsigned int)(37 * v6) | ((unsigned __int64)(37 * v7) << 32))
          - 1
          - ((unsigned __int64)(unsigned int)(37 * v6) << 32)) >> 22)
        ^ (((unsigned int)(37 * v6) | ((unsigned __int64)(37 * v7) << 32))
         - 1
         - ((unsigned __int64)(unsigned int)(37 * v6) << 32));
    v14 = ((9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13)))) >> 15)
        ^ (9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13))));
    for ( i = v11 & (((v14 - 1 - (v14 << 27)) >> 31) ^ (v14 - 1 - ((_DWORD)v14 << 27))); ; i = v11 & v17 )
    {
      v16 = (_DWORD *)(v10 + 48LL * i);
      if ( *v16 == v7 && v16[1] == (_DWORD)v6 )
        break;
      if ( *v16 == -1 && v16[1] == -1 )
        goto LABEL_11;
      v17 = v12 + i;
      ++v12;
    }
    v46 = 0x200000000LL;
    v22 = (unsigned int)v16[4];
    v45 = v47;
    if ( !(_DWORD)v22 )
    {
      v19 = v47;
LABEL_12:
      v20 = (v6 << 32) | v7;
      if ( v19 == v47 )
        return v20;
      goto LABEL_13;
    }
    v37 = a5;
    v39 = v6;
    v42 = v10 + 48LL * i;
    sub_1E9C380((__int64)&v45, (__int64)(v16 + 2), v22, v6, a5, (int)v16);
    v23 = v46;
    v19 = v45;
    v6 = v39;
    a5 = v37;
    v48 = *(_QWORD *)(v42 + 40);
    if ( (int)v46 <= 0 )
      goto LABEL_12;
    if ( (_DWORD)v46 != 1 )
      break;
    v7 = *(_DWORD *)v45;
    v6 = *((unsigned int *)v45 + 1);
    if ( v45 != v47 )
    {
      v43 = *((_DWORD *)v45 + 1);
      _libc_free((unsigned __int64)v45);
      v6 = v43;
      a5 = v37;
    }
  }
  if ( (_BYTE)v37 )
  {
    v24 = 0;
    v53 = (__int32 *)v55;
    v54 = 0x400000000LL;
    while ( 1 )
    {
      v25 = sub_1E9D410(a1, a2, *(_QWORD *)&v19[8 * v24], a4, 1);
      v27 = (unsigned int)v54;
      if ( (unsigned int)v54 >= HIDWORD(v54) )
      {
        v41 = v25;
        sub_16CD150((__int64)&v53, v55, 0, 8, v25, v26);
        v27 = (unsigned int)v54;
        v25 = v41;
      }
      ++v24;
      *(_QWORD *)&v53[2 * v27] = v25;
      LODWORD(v54) = v54 + 1;
      if ( v23 <= (unsigned int)v24 )
        break;
      v19 = v45;
    }
    v28 = v48;
    v38 = v48;
    v29 = sub_1E6B9A0(
            a1,
            *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * (*v53 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
            (unsigned __int8 *)byte_3F871B3,
            0,
            v25,
            v26);
    v30 = sub_1E9D330(*(_QWORD *)(v28 + 24), v28, (__int64 *)(v28 + 64), *(_QWORD *)(a2 + 8), v29);
    v32 = v31;
    v40 = &v53[2 * (unsigned int)v54];
    if ( v53 != v40 )
    {
      v44 = 2;
      v33 = v53;
      do
      {
        v34 = v33[1];
        v35 = *v33;
        v33 += 2;
        v49.m128i_i32[2] = v35;
        v50 = 0;
        v49.m128i_i64[0] = (unsigned __int16)(v34 & 0xFFF) << 8;
        v51 = 0;
        v52 = 0;
        sub_1E1A9C0(v32, v30, &v49);
        v36 = *(_QWORD *)(*(_QWORD *)(v38 + 32) + 40LL * v44 + 24);
        v49.m128i_i8[0] = 4;
        v50 = 0;
        v51 = v36;
        v49.m128i_i32[0] &= 0xFFF000FF;
        sub_1E1A9C0(v32, v30, &v49);
        sub_1E69E80(a1, *(v33 - 2));
        v44 += 2;
      }
      while ( v40 != v33 );
      v40 = v53;
    }
    v20 = ((unsigned __int64)((**(_DWORD **)(v32 + 32) >> 8) & 0xFFF) << 32)
        | *(unsigned int *)(*(_QWORD *)(v32 + 32) + 8LL);
    if ( v40 != (__int32 *)v55 )
      _libc_free((unsigned __int64)v40);
    v19 = v45;
    if ( v45 == v47 )
      return v20;
LABEL_13:
    _libc_free((unsigned __int64)v19);
    return v20;
  }
  if ( v45 != v47 )
    _libc_free((unsigned __int64)v45);
  return 0;
}
