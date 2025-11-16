// Function: sub_30389E0
// Address: 0x30389e0
//
__int64 __fastcall sub_30389E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rax
  int v10; // ebx
  __int16 *v11; // rdx
  unsigned __int16 v12; // ax
  __int64 v13; // rdx
  int v14; // r12d
  unsigned int v15; // eax
  int v16; // eax
  __int64 v17; // r12
  int v18; // r13d
  __int128 v19; // rax
  int v20; // r9d
  __int64 v21; // rax
  __int64 v22; // r10
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r11
  unsigned __int64 v26; // rdx
  __int64 *v27; // rax
  __int128 v28; // rdi
  __int64 v29; // r12
  int v31; // edx
  __int64 v32; // [rsp+10h] [rbp-130h]
  __int64 v34; // [rsp+20h] [rbp-120h]
  __int64 v35; // [rsp+28h] [rbp-118h]
  __int64 v36; // [rsp+38h] [rbp-108h]
  __int64 v37; // [rsp+40h] [rbp-100h]
  unsigned __int64 v38; // [rsp+48h] [rbp-F8h]
  __m128i v39; // [rsp+50h] [rbp-F0h]
  __int64 v40; // [rsp+60h] [rbp-E0h] BYREF
  int v41; // [rsp+68h] [rbp-D8h]
  unsigned __int16 v42; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v43; // [rsp+78h] [rbp-C8h]
  _BYTE *v44; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+88h] [rbp-B8h]
  _BYTE v46[176]; // [rsp+90h] [rbp-B0h] BYREF

  v7 = a4;
  v8 = *(_QWORD *)(a2 + 80);
  v40 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v40, v8, 1);
  v41 = *(_DWORD *)(a2 + 72);
  v44 = v46;
  v45 = 0x800000000LL;
  v9 = *(unsigned int *)(a2 + 64);
  if ( (_DWORD)v9 )
  {
    HIWORD(v10) = HIWORD(v6);
    v36 = 0;
    v32 = 40 * v9;
    while ( 1 )
    {
      v11 = *(__int16 **)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + v36) + 48LL);
      v38 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + v36);
      v39 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v36));
      v12 = *v11;
      v13 = *((_QWORD *)v11 + 1);
      v42 = v12;
      v43 = v13;
      if ( v12 )
        break;
      v10 = sub_3009970((__int64)&v42, v8, v13, a4, a5);
      v12 = v42;
      v14 = v31;
      if ( v42 )
        goto LABEL_7;
      if ( !sub_3007100((__int64)&v42) )
        goto LABEL_23;
LABEL_24:
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      if ( !v42 )
      {
LABEL_23:
        v15 = sub_3007130((__int64)&v42, v8);
        goto LABEL_9;
      }
      if ( (unsigned __int16)(v42 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
LABEL_8:
      v15 = word_4456340[v42 - 1];
LABEL_9:
      v37 = v15;
      if ( v15 )
      {
        v16 = v14;
        v17 = 0;
        v18 = v16;
        do
        {
          *(_QWORD *)&v19 = sub_3400D50(v7, v17, &v40, 0);
          v8 = 158;
          v21 = sub_3406EB0(v7, 158, (unsigned int)&v40, v10, v18, v20, __PAIR128__(v39.m128i_u64[1], v38), v19);
          a4 = HIDWORD(v45);
          v22 = v21;
          v23 = (unsigned int)v45;
          v25 = v24;
          v26 = (unsigned int)v45 + 1LL;
          if ( v26 > HIDWORD(v45) )
          {
            v8 = (__int64)v46;
            v34 = v22;
            v35 = v25;
            sub_C8D5F0((__int64)&v44, v46, v26, 0x10u, a5, a6);
            v23 = (unsigned int)v45;
            v22 = v34;
            v25 = v35;
          }
          v27 = (__int64 *)&v44[16 * v23];
          ++v17;
          *v27 = v22;
          v27[1] = v25;
          LODWORD(v45) = v45 + 1;
        }
        while ( v17 != v37 );
      }
      v36 += 40;
      if ( v32 == v36 )
      {
        *(_QWORD *)&v28 = v44;
        *((_QWORD *)&v28 + 1) = (unsigned int)v45;
        goto LABEL_16;
      }
    }
    a4 = (__int64)word_4456580;
    v14 = 0;
    LOWORD(v10) = word_4456580[v12 - 1];
LABEL_7:
    if ( (unsigned __int16)(v12 - 176) > 0x34u )
      goto LABEL_8;
    goto LABEL_24;
  }
  *(_QWORD *)&v28 = v46;
  *((_QWORD *)&v28 + 1) = 0;
LABEL_16:
  v29 = sub_33FC220(
          v7,
          156,
          (unsigned int)&v40,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          a6,
          v28);
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
  return v29;
}
