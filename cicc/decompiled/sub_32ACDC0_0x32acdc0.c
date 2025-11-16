// Function: sub_32ACDC0
// Address: 0x32acdc0
//
__int64 __fastcall sub_32ACDC0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // r15
  __int64 result; // rax
  __int64 v5; // r9
  __int64 v8; // rax
  unsigned __int16 v9; // dx
  unsigned int v10; // ebx
  unsigned int v11; // r13d
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // edi
  int v20; // ecx
  unsigned int v21; // eax
  __int64 v22; // rdx
  unsigned int v23; // ebx
  __int64 v24; // rdx
  __int64 v25; // r13
  __int64 *v26; // rax
  __int64 v27; // r10
  __int64 v28; // r11
  __int64 v29; // rdx
  __int128 v30; // rax
  int v31; // r9d
  __int64 v32; // [rsp+8h] [rbp-138h]
  __int64 v33; // [rsp+10h] [rbp-130h]
  int v34; // [rsp+10h] [rbp-130h]
  int v35; // [rsp+18h] [rbp-128h]
  __int64 v36; // [rsp+30h] [rbp-110h]
  int v37; // [rsp+30h] [rbp-110h]
  __int128 v38; // [rsp+30h] [rbp-110h]
  int v39; // [rsp+38h] [rbp-108h]
  __m128i v40; // [rsp+40h] [rbp-100h]
  __int64 v41; // [rsp+40h] [rbp-100h]
  __int64 v42; // [rsp+40h] [rbp-100h]
  unsigned __int16 v43[4]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v44; // [rsp+58h] [rbp-E8h]
  __int64 v45; // [rsp+60h] [rbp-E0h] BYREF
  int v46; // [rsp+68h] [rbp-D8h]
  _BYTE *v47; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v48; // [rsp+78h] [rbp-C8h]
  _BYTE v49[64]; // [rsp+80h] [rbp-C0h] BYREF
  _BYTE *v50; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v51; // [rsp+C8h] [rbp-78h]
  _BYTE v52[112]; // [rsp+D0h] [rbp-70h] BYREF

  v2 = *(_QWORD **)(a1 + 40);
  v3 = *v2;
  if ( *(_DWORD *)(*v2 + 24LL) != 159 )
    return 0;
  if ( *(_DWORD *)(v3 + 64) != 2 )
    return 0;
  v5 = v2[5];
  if ( *(_DWORD *)(v5 + 24) != 159
    || *(_DWORD *)(v5 + 64) != 2
    || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 40) + 40LL) + 24LL) != 51
    || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL) + 24LL) != 51 )
  {
    return 0;
  }
  v32 = v2[5];
  v33 = sub_3288400(a1, a2);
  v8 = *(_QWORD *)(a1 + 48);
  v9 = *(_WORD *)v8;
  v44 = *(_QWORD *)(v8 + 8);
  v43[0] = v9;
  v10 = sub_3281500(v43, a2);
  v11 = v10 >> 1;
  v47 = v49;
  v48 = 0x1000000000LL;
  sub_11B1960((__int64)&v47, v10 >> 1, -1, v12, v13, v14);
  v15 = v10 >> 1;
  v50 = v52;
  v51 = 0x1000000000LL;
  sub_11B1960((__int64)&v50, v15, -1, v16, v17, v18);
  if ( v10 )
  {
    v15 = 0;
    v19 = -v11;
    do
    {
      v20 = *(_DWORD *)(v33 + 4 * v15);
      if ( v20 != -1 && v20 % v10 < v11 )
      {
        if ( v20 >= (int)v10 )
          v20 -= v11;
        if ( v11 <= (unsigned int)v15 )
          *(_DWORD *)&v50[4 * v19] = v20;
        else
          *(_DWORD *)&v47[4 * v15] = v20;
      }
      ++v15;
      ++v19;
    }
    while ( v15 != v10 );
  }
  v36 = *(_QWORD *)(a2 + 16);
  LOWORD(v21) = sub_3281100(v43, v15);
  v23 = sub_327FCF0(*(__int64 **)(a2 + 64), v21, v22, v11, 0);
  v25 = v24;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _BYTE *, _QWORD, _QWORD, __int64))(*(_QWORD *)v36 + 624LL))(
         v36,
         v47,
         (unsigned int)v48,
         v23,
         v24)
    && (*(unsigned __int8 (__fastcall **)(__int64, _BYTE *, _QWORD, _QWORD, __int64))(*(_QWORD *)v36 + 624LL))(
         v36,
         v50,
         (unsigned int)v51,
         v23,
         v25) )
  {
    v26 = *(__int64 **)(v3 + 40);
    v27 = *v26;
    v28 = v26[1];
    v40 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v32 + 40));
    v45 = *(_QWORD *)(a1 + 80);
    if ( v45 )
    {
      v37 = v27;
      v39 = v28;
      sub_325F5D0(&v45);
      LODWORD(v27) = v37;
      LODWORD(v28) = v39;
    }
    v34 = v27;
    v46 = *(_DWORD *)(a1 + 72);
    v35 = v28;
    *(_QWORD *)&v38 = sub_33FCE10(
                        a2,
                        v23,
                        v25,
                        (unsigned int)&v45,
                        v27,
                        v28,
                        v40.m128i_i64[0],
                        v40.m128i_i64[1],
                        (__int64)v47,
                        (unsigned int)v48);
    *((_QWORD *)&v38 + 1) = v29;
    *(_QWORD *)&v30 = sub_33FCE10(
                        a2,
                        v23,
                        v25,
                        (unsigned int)&v45,
                        v34,
                        v35,
                        v40.m128i_i64[0],
                        v40.m128i_i64[1],
                        (__int64)v50,
                        (unsigned int)v51);
    *(_QWORD *)&v38 = sub_3406EB0(a2, 159, (unsigned int)&v45, *(_DWORD *)v43, v44, v31, v38, v30);
    sub_9C6650(&v45);
    result = v38;
  }
  else
  {
    result = 0;
  }
  if ( v50 != v52 )
  {
    v41 = result;
    _libc_free((unsigned __int64)v50);
    result = v41;
  }
  if ( v47 != v49 )
  {
    v42 = result;
    _libc_free((unsigned __int64)v47);
    return v42;
  }
  return result;
}
