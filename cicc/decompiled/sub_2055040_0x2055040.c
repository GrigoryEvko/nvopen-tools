// Function: sub_2055040
// Address: 0x2055040
//
__int64 *__fastcall sub_2055040(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        double a6,
        double a7,
        __m128i a8)
{
  __int64 *v10; // r12
  __int64 v11; // rsi
  unsigned int v13; // edx
  int v14; // eax
  unsigned int v15; // esi
  int v16; // edx
  unsigned __int64 v17; // rax
  int v18; // eax
  unsigned int v19; // eax
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdi
  const void ***v26; // rdx
  __m128i v27; // rax
  __int64 v28; // rcx
  int v29; // r8d
  int v30; // r9d
  int v31; // r13d
  unsigned int v32; // r12d
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // rax
  __int64 v36; // r14
  _QWORD *v37; // rax
  __int128 v38; // [rsp-10h] [rbp-120h]
  unsigned __int64 v39; // [rsp+8h] [rbp-108h]
  unsigned int v40; // [rsp+18h] [rbp-F8h]
  unsigned int v41; // [rsp+18h] [rbp-F8h]
  unsigned int v42; // [rsp+18h] [rbp-F8h]
  unsigned int v43; // [rsp+20h] [rbp-F0h]
  __int64 v45; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v46; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v47; // [rsp+40h] [rbp-D0h] BYREF
  unsigned int v48; // [rsp+48h] [rbp-C8h]
  __int64 v49[2]; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v50; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v51; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v52; // [rsp+78h] [rbp-98h]
  __int64 v53; // [rsp+80h] [rbp-90h]
  unsigned int v54; // [rsp+88h] [rbp-88h]
  _BYTE *v55; // [rsp+90h] [rbp-80h] BYREF
  __int64 v56; // [rsp+98h] [rbp-78h]
  _BYTE v57[112]; // [rsp+A0h] [rbp-70h] BYREF

  v10 = (__int64 *)a4;
  v43 = a5;
  if ( !*(_QWORD *)(a3 + 48) && *(__int16 *)(a3 + 18) >= 0 )
    return v10;
  v11 = sub_1625790(a3, 4);
  if ( !v11 )
    return v10;
  sub_1593050((__int64)&v51, v11);
  if ( !sub_158A0B0((__int64)&v51) && !sub_158A120((__int64)&v51) && !sub_158A670((__int64)&v51) )
  {
    sub_158AAD0((__int64)&v45, (__int64)&v51);
    v13 = v46;
    if ( v46 <= 0x40 )
    {
      if ( v45 )
        goto LABEL_17;
    }
    else
    {
      v40 = v46;
      v14 = sub_16A57B0((__int64)&v45);
      v13 = v40;
      if ( v40 != v14 )
      {
LABEL_17:
        if ( v13 > 0x40 && v45 )
          j_j___libc_free_0_0(v45);
        goto LABEL_4;
      }
    }
    sub_158A9F0((__int64)&v47, (__int64)&v51);
    v15 = v48;
    if ( v48 > 0x40 )
    {
      v42 = v48;
      v18 = sub_16A57B0((__int64)&v47);
      v15 = v42;
    }
    else
    {
      v16 = 64;
      if ( v47 )
      {
        _BitScanReverse64(&v17, v47);
        v16 = v17 ^ 0x3F;
      }
      v18 = v48 + v16 - 64;
    }
    v19 = sub_1F7DE30(*(_QWORD **)(a2 + 48), v15 - v18);
    v39 = v20;
    v41 = v19;
    sub_204D410((__int64)v49, *(_QWORD *)a1, *(_DWORD *)(a1 + 536));
    v23 = sub_1D2EF30((_QWORD *)a2, v41, v39, v21, v39, v22);
    v25 = v24;
    v26 = (const void ***)(*(_QWORD *)(a4 + 40) + 16LL * v43);
    *((_QWORD *)&v38 + 1) = v25;
    *(_QWORD *)&v38 = v23;
    v27.m128i_i64[0] = (__int64)sub_1D332F0(
                                  (__int64 *)a2,
                                  4,
                                  (__int64)v49,
                                  *(unsigned __int8 *)v26,
                                  v26[1],
                                  0,
                                  a6,
                                  a7,
                                  a8,
                                  (__int64)v10,
                                  a5,
                                  v38);
    v31 = *(_DWORD *)(a4 + 60);
    v50 = v27;
    if ( v31 == 1 )
    {
      v10 = (__int64 *)v27.m128i_i64[0];
    }
    else
    {
      v32 = 1;
      v56 = 0x400000000LL;
      v55 = v57;
      sub_1D23890((__int64)&v55, &v50, v27.m128i_i64[1], v28, v29, v30);
      v35 = (unsigned int)v56;
      do
      {
        v36 = v32;
        if ( (unsigned int)v35 >= HIDWORD(v56) )
        {
          sub_16CD150((__int64)&v55, v57, 0, 16, v33, v34);
          v35 = (unsigned int)v56;
        }
        v37 = &v55[16 * v35];
        ++v32;
        *v37 = a4;
        v37[1] = v36;
        v35 = (unsigned int)(v56 + 1);
        LODWORD(v56) = v56 + 1;
      }
      while ( v31 != v32 );
      v10 = sub_1D37190(a2, (__int64)v55, (unsigned int)v35, (__int64)v49, v33, a6, a7, a8);
      if ( v55 != v57 )
        _libc_free((unsigned __int64)v55);
    }
    if ( v49[0] )
      sub_161E7C0((__int64)v49, v49[0]);
    if ( v48 > 0x40 && v47 )
      j_j___libc_free_0_0(v47);
    v13 = v46;
    goto LABEL_17;
  }
LABEL_4:
  if ( v54 > 0x40 && v53 )
    j_j___libc_free_0_0(v53);
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  return v10;
}
