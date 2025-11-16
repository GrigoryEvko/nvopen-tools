// Function: sub_32EA290
// Address: 0x32ea290
//
__int64 __fastcall sub_32EA290(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r13
  __int64 v8; // r12
  unsigned __int16 *v9; // rax
  __int64 v10; // rsi
  unsigned __int16 v11; // r15
  __int64 v12; // r14
  bool v13; // zf
  __int64 v14; // rdi
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned __int16 *v20; // rdx
  const __m128i *v21; // roff
  __int64 v22; // rbx
  unsigned __int32 v23; // r12d
  int v24; // eax
  __int64 v25; // rdx
  char v26; // r11
  unsigned __int64 v27; // r10
  unsigned __int64 v28; // rsi
  unsigned __int16 *v29; // rdx
  int v30; // eax
  __int64 v31; // rdx
  unsigned __int16 v32; // ax
  unsigned int v33; // r13d
  __int16 v34; // si
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  int v38; // edx
  __int64 v39; // rdx
  __int64 v40; // rdx
  unsigned int *v41; // rdx
  __int64 v42; // rax
  unsigned __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+10h] [rbp-A0h]
  __int64 *v48; // [rsp+18h] [rbp-98h]
  char v49; // [rsp+18h] [rbp-98h]
  __m128i v50; // [rsp+20h] [rbp-90h]
  __int128 v51; // [rsp+20h] [rbp-90h]
  char v52; // [rsp+32h] [rbp-7Eh]
  char v53; // [rsp+33h] [rbp-7Dh]
  unsigned int v54; // [rsp+34h] [rbp-7Ch]
  __int64 v55; // [rsp+38h] [rbp-78h]
  __int64 v56; // [rsp+40h] [rbp-70h]
  __int64 v57; // [rsp+50h] [rbp-60h] BYREF
  int v58; // [rsp+58h] [rbp-58h]
  int v59; // [rsp+60h] [rbp-50h] BYREF
  __int64 v60; // [rsp+68h] [rbp-48h]
  __int64 v61; // [rsp+70h] [rbp-40h] BYREF
  __int64 v62; // [rsp+78h] [rbp-38h]

  v6 = a2;
  v8 = **(_QWORD **)(a2 + 40);
  v9 = *(unsigned __int16 **)(a2 + 48);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v57 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v57, v10, 1);
  v13 = *(_DWORD *)(v8 + 24) == 51;
  v14 = *(_QWORD *)a1;
  v58 = *(_DWORD *)(v6 + 72);
  if ( v13 )
  {
    if ( *(_DWORD *)(v6 + 24) == 223 )
    {
      v61 = 0;
      LODWORD(v62) = 0;
      v6 = sub_33F17F0(v14, 51, &v61, v11, v12);
      if ( v61 )
        sub_B91220((__int64)&v61, v61);
    }
    else
    {
      v6 = sub_3400BD0(v14, 0, (unsigned int)&v57, v11, v12, 0, 0);
    }
  }
  else
  {
    v16 = sub_32788C0(v6, (int)&v57, *(_QWORD *)(a1 + 8), v14, *(_BYTE *)(a1 + 34), a6);
    v17 = v16;
    if ( v16 )
    {
      v6 = v16;
    }
    else if ( !(unsigned __int8)sub_32E2EF0(a1, v6, 0) )
    {
      v53 = *(_BYTE *)(a1 + 33);
      v55 = *(_QWORD *)a1;
      v46 = *(_QWORD *)(a1 + 8);
      switch ( *(_DWORD *)(v6 + 24) )
      {
        case 0xD5:
        case 0xE0:
          v54 = 213;
          break;
        case 0xD6:
        case 0xE1:
          v54 = 214;
          break;
        case 0xD7:
        case 0xDF:
          v54 = 215;
          break;
        default:
          BUG();
      }
      v20 = *(unsigned __int16 **)(v6 + 48);
      v21 = *(const __m128i **)(v6 + 40);
      v22 = v21->m128i_i64[0];
      v23 = v21->m128i_u32[2];
      v24 = *v20;
      v25 = *((_QWORD *)v20 + 1);
      v50 = _mm_loadu_si128(v21);
      LOWORD(v59) = v24;
      v60 = v25;
      if ( (_WORD)v24 )
      {
        v26 = (unsigned __int16)(v24 - 176) <= 0x34u;
        LOBYTE(v27) = v26;
        v28 = word_4456340[v24 - 1];
      }
      else
      {
        v28 = sub_3007240((__int64)&v59);
        v27 = HIDWORD(v28);
        v26 = BYTE4(v28);
      }
      v29 = (unsigned __int16 *)(*(_QWORD *)(v22 + 48) + 16LL * v23);
      v30 = *v29;
      v31 = *((_QWORD *)v29 + 1);
      LOWORD(v61) = v30;
      v62 = v31;
      if ( (_WORD)v30 )
      {
        v47 = 0;
        v32 = word_4456580[v30 - 1];
      }
      else
      {
        v52 = v27;
        v49 = v26;
        v32 = sub_3009970((__int64)&v61, v28, v31, v18, v19);
        LOBYTE(v27) = v52;
        v47 = v39;
        v26 = v49;
      }
      LODWORD(v56) = v28;
      v33 = v32;
      BYTE4(v56) = v27;
      v48 = *(__int64 **)(v55 + 64);
      if ( v26 )
        v34 = sub_2D43AD0(v32, v28);
      else
        v34 = sub_2D43050(v32, v28);
      if ( !v34 )
      {
        v34 = sub_3009450(v48, v33, v47, v56, v35, v36);
        v17 = v40;
      }
      v37 = *(_QWORD *)(v22 + 56);
      if ( !v37 )
        goto LABEL_36;
      v38 = 1;
      do
      {
        if ( *(_DWORD *)(v37 + 8) == v23 )
        {
          if ( !v38 )
            goto LABEL_36;
          v37 = *(_QWORD *)(v37 + 32);
          if ( !v37 )
            goto LABEL_35;
          if ( v23 == *(_DWORD *)(v37 + 8) )
            goto LABEL_36;
          v38 = 0;
        }
        v37 = *(_QWORD *)(v37 + 32);
      }
      while ( v37 );
      if ( v38 == 1 )
        goto LABEL_36;
LABEL_35:
      if ( *(_DWORD *)(v22 + 24) != 159 )
        goto LABEL_36;
      v41 = *(unsigned int **)(v22 + 40);
      v50.m128i_i64[0] = *(_QWORD *)v41;
      v42 = v41[2];
      v43 = v42 | v50.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v44 = *(_QWORD *)(*(_QWORD *)v41 + 48LL) + 16 * v42;
      *((_QWORD *)&v51 + 1) = v43;
      if ( *(_WORD *)v44 != v34 || *(_QWORD *)(v44 + 8) != v17 && !v34 )
        goto LABEL_36;
      if ( v53 )
      {
        v45 = 1;
        if ( (_WORD)v59 != 1 )
        {
          if ( !(_WORD)v59 )
            goto LABEL_36;
          v45 = (unsigned __int16)v59;
          if ( !*(_QWORD *)(v46 + 8LL * (unsigned __int16)v59 + 112) )
            goto LABEL_36;
        }
        if ( *(_BYTE *)(v54 + v46 + 500 * v45 + 6414) )
          goto LABEL_36;
      }
      v6 = sub_33FAF80(v55, v54, (unsigned int)&v57, v59, v60, v36, v51);
      if ( !v6 )
LABEL_36:
        v6 = 0;
    }
  }
  if ( v57 )
    sub_B91220((__int64)&v57, v57);
  return v6;
}
