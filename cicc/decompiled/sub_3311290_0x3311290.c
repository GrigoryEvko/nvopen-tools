// Function: sub_3311290
// Address: 0x3311290
//
__int64 __fastcall sub_3311290(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r13
  unsigned __int16 *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 result; // rax
  __int64 v17; // rdx
  __int64 v18; // r11
  int v19; // eax
  __int64 v20; // rax
  int v21; // edx
  _QWORD *v22; // r13
  __int64 v23; // rdx
  _QWORD *v24; // rax
  __int64 v25; // r14
  __int64 v26; // r15
  __int64 v27; // rax
  unsigned __int16 v28; // dx
  __int64 v29; // r13
  __int64 v30; // rsi
  unsigned __int16 v31; // dx
  __m128i v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rbx
  __int64 v35; // r11
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned __int16 *v38; // r13
  __int64 v39; // r8
  int v40; // ecx
  __int64 v41; // r9
  __int64 v42; // r10
  __int64 v43; // r11
  __int32 v44; // edx
  __int64 v45; // r11
  bool v46; // al
  int v47; // r9d
  __int64 v48; // rdi
  __int128 v49; // [rsp-20h] [rbp-100h]
  int v50; // [rsp-10h] [rbp-F0h]
  __int128 v51; // [rsp-10h] [rbp-F0h]
  __int128 v52; // [rsp-10h] [rbp-F0h]
  __int64 v53; // [rsp+0h] [rbp-E0h]
  int v54; // [rsp+8h] [rbp-D8h]
  __int64 v55; // [rsp+10h] [rbp-D0h]
  __int64 v56; // [rsp+18h] [rbp-C8h]
  __int64 v57; // [rsp+20h] [rbp-C0h]
  __int64 v58; // [rsp+20h] [rbp-C0h]
  int v59; // [rsp+20h] [rbp-C0h]
  __int64 v60; // [rsp+20h] [rbp-C0h]
  __int64 v61; // [rsp+28h] [rbp-B8h]
  __int64 v62; // [rsp+38h] [rbp-A8h]
  __m128i v63; // [rsp+40h] [rbp-A0h] BYREF
  int v64; // [rsp+50h] [rbp-90h] BYREF
  __int64 v65; // [rsp+58h] [rbp-88h]
  __int64 v66; // [rsp+60h] [rbp-80h] BYREF
  int v67; // [rsp+68h] [rbp-78h]
  __int64 v68; // [rsp+70h] [rbp-70h] BYREF
  int v69; // [rsp+78h] [rbp-68h]
  __int64 v70; // [rsp+80h] [rbp-60h] BYREF
  int v71; // [rsp+88h] [rbp-58h]
  __m128i v72; // [rsp+90h] [rbp-50h] BYREF
  __int64 v73; // [rsp+A0h] [rbp-40h]
  int v74; // [rsp+A8h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)v7;
  v9 = *(unsigned int *)(v7 + 8);
  v10 = *(unsigned __int16 **)(a2 + 48);
  v11 = *(_QWORD *)(a2 + 80);
  v12 = *v10;
  v13 = *((_QWORD *)v10 + 1);
  v66 = v11;
  v62 = v13;
  LOWORD(v64) = v12;
  v65 = v13;
  if ( v11 )
    sub_B96E90((__int64)&v66, v11, 1);
  v67 = *(_DWORD *)(a2 + 72);
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
      goto LABEL_5;
  }
  else if ( !sub_30070B0((__int64)&v64) )
  {
    goto LABEL_5;
  }
  result = sub_32730B0(a1, a2, (__int64)&v66, a4, a5);
  if ( result )
    goto LABEL_7;
LABEL_5:
  v14 = *(_QWORD *)(a2 + 56);
  if ( v14 && !*(_QWORD *)(v14 + 32) && *(_DWORD *)(*(_QWORD *)(v14 + 16) + 24LL) == 230 )
  {
    result = 0;
    goto LABEL_7;
  }
  v15 = *a1;
  v63.m128i_i64[0] = (__int64)&v66;
  v72.m128i_i64[0] = v8;
  v72.m128i_i32[2] = v9;
  result = sub_3402EA0(v15, 233, (unsigned int)&v66, v64, v65, 0, (__int64)&v72, 1);
  v18 = v17;
  if ( result )
    goto LABEL_7;
  v19 = *(_DWORD *)(v8 + 24);
  if ( v19 == 236 )
  {
    if ( (_WORD)v12 && !*(_BYTE *)(a1[1] + 500 * v12 + 6650) )
    {
      result = sub_33FAF80(*a1, 236, v63.m128i_i32[0], v64, v65, v50, *(_OWORD *)*(_QWORD *)(v8 + 40));
      goto LABEL_7;
    }
    goto LABEL_21;
  }
  if ( v19 == 230 )
  {
    v22 = *(_QWORD **)(v8 + 40);
    v23 = *(_QWORD *)(v22[5] + 96LL);
    v24 = *(_QWORD **)(v23 + 24);
    if ( *(_DWORD *)(v23 + 32) > 0x40u )
      v24 = (_QWORD *)*v24;
    if ( v24 == (_QWORD *)1 )
    {
      v25 = *v22;
      v26 = v22[1];
      v27 = *(_QWORD *)(*v22 + 48LL) + 16LL * *((unsigned int *)v22 + 2);
      v28 = *(_WORD *)v27;
      if ( (_WORD)v12 == *(_WORD *)v27 && (v28 || *(_QWORD *)(v27 + 8) == v62) )
      {
        result = *v22;
      }
      else
      {
        v46 = sub_3280B30((__int64)&v64, v28, *(_QWORD *)(v27 + 8));
        v48 = *a1;
        if ( v46 )
        {
          *((_QWORD *)&v49 + 1) = v26;
          *(_QWORD *)&v49 = v25;
          result = sub_3406EB0(v48, 230, v63.m128i_i32[0], v64, v65, v47, v49, *(_OWORD *)(v22 + 5));
        }
        else
        {
          *((_QWORD *)&v52 + 1) = v26;
          *(_QWORD *)&v52 = v25;
          result = sub_33FAF80(v48, 233, v63.m128i_i32[0], v64, v65, v47, v52);
        }
      }
      goto LABEL_7;
    }
LABEL_21:
    result = sub_327CF00(a1, a2);
    goto LABEL_7;
  }
  if ( v19 != 298 )
    goto LABEL_21;
  if ( (*(_BYTE *)(v8 + 33) & 0xC) != 0 )
    goto LABEL_21;
  if ( (*(_WORD *)(v8 + 32) & 0x380) != 0 )
    goto LABEL_21;
  v20 = *(_QWORD *)(v8 + 56);
  if ( !v20 )
    goto LABEL_21;
  v21 = 1;
  do
  {
    if ( *(_DWORD *)(v20 + 8) == (_DWORD)v9 )
    {
      if ( !v21 )
        goto LABEL_21;
      v20 = *(_QWORD *)(v20 + 32);
      if ( !v20 )
        goto LABEL_41;
      if ( (_DWORD)v9 == *(_DWORD *)(v20 + 8) )
        goto LABEL_21;
      v21 = 0;
    }
    v20 = *(_QWORD *)(v20 + 32);
  }
  while ( v20 );
  if ( v21 == 1 )
    goto LABEL_21;
LABEL_41:
  v29 = 16 * v9;
  v30 = v29 + *(_QWORD *)(v8 + 48);
  if ( !(_WORD)v12 )
    goto LABEL_21;
  v31 = *(_WORD *)v30;
  if ( !*(_WORD *)v30
    || (((int)*(unsigned __int16 *)(a1[1] + 2 * (v31 + 274LL * (unsigned __int16)v12 + 71704) + 6) >> 4) & 0xB) != 0 )
  {
    goto LABEL_21;
  }
  v57 = v18;
  v32.m128i_i64[0] = sub_33F1B30(
                       *a1,
                       1,
                       v63.m128i_i32[0],
                       v64,
                       v65,
                       *(_QWORD *)(v8 + 112),
                       **(_QWORD **)(v8 + 40),
                       *(_QWORD *)(*(_QWORD *)(v8 + 40) + 8LL),
                       *(_QWORD *)(*(_QWORD *)(v8 + 40) + 40LL),
                       *(_QWORD *)(*(_QWORD *)(v8 + 40) + 48LL),
                       v31,
                       *(_QWORD *)(v30 + 8));
  v63 = v32;
  v72 = _mm_load_si128(&v63);
  sub_32EB790((__int64)a1, a2, v72.m128i_i64, 1, 1);
  v33 = *(_QWORD *)(v8 + 80);
  v34 = *a1;
  v61 = v63.m128i_i64[0];
  v35 = v57;
  v70 = v33;
  if ( v33 )
  {
    sub_B96E90((__int64)&v70, v33, 1);
    v35 = v57;
  }
  v58 = v35;
  v71 = *(_DWORD *)(v8 + 72);
  v36 = sub_3400D50(v34, 1, &v70, 1);
  v38 = (unsigned __int16 *)(*(_QWORD *)(v8 + 48) + v29);
  v39 = *((_QWORD *)v38 + 1);
  v40 = *v38;
  v41 = v36;
  v42 = v37;
  v68 = *(_QWORD *)(v8 + 80);
  v43 = v58;
  if ( v68 )
  {
    v56 = v37;
    v53 = v58;
    v54 = v40;
    v59 = v39;
    v55 = v36;
    sub_B96E90((__int64)&v68, v68, 1);
    v43 = v53;
    v40 = v54;
    v41 = v55;
    v42 = v56;
    LODWORD(v39) = v59;
  }
  *((_QWORD *)&v51 + 1) = v42;
  *(_QWORD *)&v51 = v41;
  v60 = v43;
  v69 = *(_DWORD *)(v8 + 72);
  v72.m128i_i64[0] = sub_3406EB0(v34, 230, (unsigned int)&v68, v40, v39, v41, *(_OWORD *)&v63, v51);
  v72.m128i_i32[2] = v44;
  v73 = v61;
  v74 = 1;
  sub_32EB790((__int64)a1, v8, v72.m128i_i64, 2, 1);
  v45 = v60;
  if ( v68 )
  {
    v63.m128i_i64[0] = v60;
    sub_B91220((__int64)&v68, v68);
    v45 = v63.m128i_i64[0];
  }
  if ( v70 )
  {
    v63.m128i_i64[0] = v45;
    sub_B91220((__int64)&v70, v70);
  }
  result = a2;
LABEL_7:
  if ( v66 )
  {
    v63.m128i_i64[0] = result;
    sub_B91220((__int64)&v66, v66);
    return v63.m128i_i64[0];
  }
  return result;
}
