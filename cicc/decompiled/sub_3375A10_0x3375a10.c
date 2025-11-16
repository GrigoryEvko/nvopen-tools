// Function: sub_3375A10
// Address: 0x3375a10
//
__int64 __fastcall sub_3375A10(__int64 a1, __int64 a2, unsigned __int8 *a3, unsigned __int64 a4, __int64 a5)
{
  unsigned int v6; // r13d
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  char v11; // al
  __int64 v12; // r12
  unsigned int v14; // ecx
  int v15; // eax
  unsigned int v16; // esi
  int v17; // edx
  unsigned __int64 v18; // rax
  int v19; // eax
  unsigned int v20; // esi
  unsigned int v21; // r15d
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdi
  unsigned __int16 *v26; // rdx
  int v27; // r9d
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r9
  __int64 v32; // r8
  unsigned int v33; // r13d
  __int64 v34; // rax
  __int64 v35; // r8
  unsigned __int64 v36; // r12
  unsigned __int64 *v37; // rax
  __int128 v38; // [rsp-10h] [rbp-130h]
  __int64 v39; // [rsp+8h] [rbp-118h]
  unsigned int v40; // [rsp+8h] [rbp-118h]
  unsigned int v41; // [rsp+8h] [rbp-118h]
  unsigned int v42; // [rsp+10h] [rbp-110h]
  unsigned int v43; // [rsp+10h] [rbp-110h]
  unsigned __int64 v44; // [rsp+18h] [rbp-108h]
  __int128 v47; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v48; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v49; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v50; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v51; // [rsp+58h] [rbp-C8h]
  __int64 v52[2]; // [rsp+60h] [rbp-C0h] BYREF
  unsigned __int64 *v53; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v54; // [rsp+78h] [rbp-A8h]
  unsigned __int64 v55; // [rsp+80h] [rbp-A0h]
  unsigned int v56; // [rsp+88h] [rbp-98h]
  char v57; // [rsp+90h] [rbp-90h]
  unsigned __int64 *v58; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v59; // [rsp+A8h] [rbp-78h]
  unsigned __int64 v60; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v61; // [rsp+B8h] [rbp-68h]

  *(_QWORD *)&v47 = a4;
  *((_QWORD *)&v47 + 1) = a5;
  v6 = a5;
  v7 = (unsigned int)*a3 - 34;
  if ( (unsigned __int8)(*a3 - 34) <= 0x33u
    && (v8 = 0x8000000000041LL, _bittest64(&v8, v7))
    && ((unsigned __int8)sub_A74710((_QWORD *)a3 + 9, 0, 40)
     || (v9 = *((_QWORD *)a3 - 4)) != 0
     && !*(_BYTE *)v9
     && *(_QWORD *)(v9 + 24) == *((_QWORD *)a3 + 10)
     && (v58 = *(unsigned __int64 **)(v9 + 120), (unsigned __int8)sub_A74710(&v58, 0, 40))) )
  {
    sub_B492D0((__int64)&v53, (__int64)a3);
    if ( !v57 )
      return v47;
  }
  else
  {
    if ( (a3[7] & 0x20) == 0 )
      return v47;
    if ( !sub_B91C10((__int64)a3, 29) )
      return v47;
    if ( (a3[7] & 0x20) == 0 )
      return v47;
    v10 = sub_B91C10((__int64)a3, 4);
    if ( !v10 )
      return v47;
    sub_ABEA30((__int64)&v58, v10);
    v57 = 1;
    v54 = v59;
    v53 = v58;
    v56 = v61;
    v55 = v60;
  }
  if ( !sub_AAF760((__int64)&v53) && !sub_AAF7D0((__int64)&v53) && !sub_AB0100((__int64)&v53) )
  {
    sub_AB0A00((__int64)&v48, (__int64)&v53);
    v14 = v49;
    if ( v49 <= 0x40 )
    {
      if ( v48 )
        goto LABEL_29;
    }
    else
    {
      v42 = v49;
      v15 = sub_C444A0((__int64)&v48);
      v14 = v42;
      if ( v42 != v15 )
      {
LABEL_29:
        v12 = v47;
LABEL_30:
        if ( v14 > 0x40 && v48 )
          j_j___libc_free_0_0(v48);
        v11 = v57;
        goto LABEL_14;
      }
    }
    sub_AB0910((__int64)&v50, (__int64)&v53);
    v16 = v51;
    if ( v51 > 0x40 )
    {
      v43 = v51;
      v19 = sub_C444A0((__int64)&v50);
      v16 = v43;
    }
    else
    {
      v17 = 64;
      if ( v50 )
      {
        _BitScanReverse64(&v18, v50);
        v17 = v18 ^ 0x3F;
      }
      v19 = v51 + v17 - 64;
    }
    v20 = v16 - v19;
    if ( !v20 )
      v20 = 1;
    v21 = sub_327FC40(*(_QWORD **)(a2 + 64), v20);
    v39 = v22;
    sub_336E8F0((__int64)v52, *(_QWORD *)a1, *(_DWORD *)(a1 + 848));
    v23 = sub_33F7D60(a2, v21, v39);
    v25 = v24;
    v26 = (unsigned __int16 *)(*(_QWORD *)(a4 + 48) + 16LL * v6);
    *((_QWORD *)&v38 + 1) = v25;
    *(_QWORD *)&v38 = v23;
    v28 = sub_3406EB0(a2, 4, (unsigned int)v52, *v26, *((_QWORD *)v26 + 1), v27, v47, v38);
    v32 = *(unsigned int *)(a4 + 68);
    v12 = v28;
    if ( (_DWORD)v32 != 1 )
    {
      v33 = 1;
      v40 = *(_DWORD *)(a4 + 68);
      v59 = 0x400000000LL;
      v58 = &v60;
      sub_3050D50((__int64)&v58, v28, v29, v30, v32, v31);
      v34 = (unsigned int)v59;
      v35 = v40;
      do
      {
        v36 = v33 | v44 & 0xFFFFFFFF00000000LL;
        v44 = v36;
        if ( v34 + 1 > (unsigned __int64)HIDWORD(v59) )
        {
          v41 = v35;
          sub_C8D5F0((__int64)&v58, &v60, v34 + 1, 0x10u, v35, 0xFFFFFFFF00000000LL);
          v34 = (unsigned int)v59;
          v35 = v41;
        }
        v37 = &v58[2 * v34];
        ++v33;
        v37[1] = v36;
        *v37 = a4;
        v34 = (unsigned int)(v59 + 1);
        LODWORD(v59) = v59 + 1;
      }
      while ( (_DWORD)v35 != v33 );
      v12 = sub_3411660(a2, v58, (unsigned int)v34, v52);
      if ( v58 != &v60 )
        _libc_free((unsigned __int64)v58);
    }
    if ( v52[0] )
      sub_B91220((__int64)v52, v52[0]);
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    v14 = v49;
    goto LABEL_30;
  }
  v11 = v57;
  v12 = v47;
LABEL_14:
  if ( v11 )
  {
    v57 = 0;
    if ( v56 > 0x40 && v55 )
      j_j___libc_free_0_0(v55);
    if ( v54 > 0x40 && v53 )
      j_j___libc_free_0_0((unsigned __int64)v53);
  }
  return v12;
}
