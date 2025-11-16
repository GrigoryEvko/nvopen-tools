// Function: sub_3030470
// Address: 0x3030470
//
__int64 __fastcall sub_3030470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // r12
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  unsigned __int16 v16; // cx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  int v21; // edx
  unsigned int v22; // eax
  __int64 v23; // rbx
  __int64 *v24; // rax
  __int128 v25; // rax
  int v26; // r9d
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rdx
  _BYTE *v30; // rdi
  __int64 v31; // rsi
  __int64 v32; // r12
  __int128 v34; // [rsp-20h] [rbp-2D0h]
  __int64 v35; // [rsp+0h] [rbp-2B0h]
  __int64 v36; // [rsp+8h] [rbp-2A8h]
  __int64 v38; // [rsp+20h] [rbp-290h]
  int v39; // [rsp+28h] [rbp-288h]
  __int64 v40; // [rsp+30h] [rbp-280h]
  __int64 v41; // [rsp+38h] [rbp-278h]
  __int64 v42; // [rsp+40h] [rbp-270h]
  __int64 v43; // [rsp+48h] [rbp-268h]
  __int64 v44; // [rsp+50h] [rbp-260h] BYREF
  int v45; // [rsp+58h] [rbp-258h]
  unsigned __int16 v46; // [rsp+60h] [rbp-250h] BYREF
  __int64 v47; // [rsp+68h] [rbp-248h]
  _BYTE *v48; // [rsp+70h] [rbp-240h] BYREF
  __int64 v49; // [rsp+78h] [rbp-238h]
  _BYTE v50[560]; // [rsp+80h] [rbp-230h] BYREF

  v7 = *(_QWORD *)(a1 + 80);
  v44 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v44, v7, 1);
  v45 = *(_DWORD *)(a1 + 72);
  v48 = v50;
  v49 = 0x2000000000LL;
  if ( *(_DWORD *)(a1 + 64) )
  {
    v43 = 0;
    while ( 1 )
    {
      v11 = a1;
      v12 = (_QWORD *)(*(_QWORD *)(a1 + 40) + 40 * v43);
      v13 = *v12;
      v14 = v12[1];
      v42 = *v12;
      v41 = *((unsigned int *)v12 + 2);
      v15 = *(_QWORD *)(*v12 + 48LL) + 16 * v41;
      v16 = *(_WORD *)v15;
      v17 = *(_QWORD *)(v15 + 8);
      v46 = v16;
      v47 = v17;
      if ( v16 )
        break;
      if ( !sub_30070B0((__int64)&v46) )
      {
LABEL_6:
        v8 = (unsigned int)v49;
        v9 = (unsigned int)v49 + 1LL;
        if ( v9 > HIDWORD(v49) )
        {
          sub_C8D5F0((__int64)&v48, v50, v9, 0x10u, a5, a6);
          v8 = (unsigned int)v49;
        }
        v10 = &v48[16 * v8];
        *v10 = v13;
        v10[1] = v14;
        LODWORD(v49) = v49 + 1;
        goto LABEL_9;
      }
      v20 = sub_3009970((__int64)&v46, a1, v18, v19, a5);
      v16 = v46;
      v40 = v20;
      v39 = v21;
      if ( v46 )
        goto LABEL_21;
      if ( !sub_3007100((__int64)&v46) )
        goto LABEL_14;
LABEL_29:
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      if ( !v46 )
      {
LABEL_14:
        v22 = sub_3007130((__int64)&v46, v11);
        goto LABEL_15;
      }
      if ( (unsigned __int16)(v46 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
LABEL_22:
      v22 = word_4456340[v46 - 1];
LABEL_15:
      v23 = 0;
      v38 = v22;
      if ( v22 )
      {
        do
        {
          *(_QWORD *)&v25 = sub_3400D50(a2, v23, &v44, 0);
          v14 = v41 | v14 & 0xFFFFFFFF00000000LL;
          *((_QWORD *)&v34 + 1) = v14;
          *(_QWORD *)&v34 = v42;
          a5 = sub_3406EB0(a2, 158, (unsigned int)&v44, v40, v39, v26, v34, v25);
          v27 = (unsigned int)v49;
          a6 = v28;
          v29 = (unsigned int)v49 + 1LL;
          if ( v29 > HIDWORD(v49) )
          {
            v35 = a5;
            v36 = a6;
            sub_C8D5F0((__int64)&v48, v50, v29, 0x10u, a5, a6);
            v27 = (unsigned int)v49;
            a5 = v35;
            a6 = v36;
          }
          v24 = (__int64 *)&v48[16 * v27];
          ++v23;
          *v24 = a5;
          v24[1] = a6;
          LODWORD(v49) = v49 + 1;
        }
        while ( v23 != v38 );
      }
LABEL_9:
      if ( *(unsigned int *)(a1 + 64) <= (unsigned __int64)++v43 )
      {
        v30 = v48;
        v31 = (unsigned int)v49;
        goto LABEL_24;
      }
    }
    if ( (unsigned __int16)(v16 - 17) > 0xD3u )
      goto LABEL_6;
    v11 = v40;
    v39 = 0;
    LOWORD(v11) = word_4456580[v16 - 1];
    v40 = v11;
LABEL_21:
    if ( (unsigned __int16)(v16 - 176) > 0x34u )
      goto LABEL_22;
    goto LABEL_29;
  }
  v30 = v50;
  v31 = 0;
LABEL_24:
  v32 = sub_33EA9D0(
          a2,
          48,
          (unsigned int)&v44,
          *(_QWORD *)(a1 + 48),
          *(_DWORD *)(a1 + 68),
          *(_QWORD *)(a1 + 112),
          (__int64)v30,
          v31,
          *(unsigned __int16 *)(a1 + 96),
          *(_QWORD *)(a1 + 104));
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  return v32;
}
