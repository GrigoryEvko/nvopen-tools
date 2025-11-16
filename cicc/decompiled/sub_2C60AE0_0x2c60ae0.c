// Function: sub_2C60AE0
// Address: 0x2c60ae0
//
__int64 __fastcall sub_2C60AE0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned int v4; // r13d
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rdx
  _BYTE *v9; // r15
  __int64 v10; // rsi
  __int64 v11; // rdi
  char v12; // dl
  __m128i v13; // rax
  _QWORD *v14; // rbx
  _BYTE *v15; // r12
  _QWORD *v16; // r13
  int v17; // ebx
  unsigned __int8 *v18; // rsi
  __m128i v19; // xmm2
  __m128i v20; // xmm0
  __m128i v21; // xmm1
  _BYTE *v22; // r13
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  char v27; // al
  _QWORD *v28; // rax
  __int64 v29; // r9
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rbx
  __int64 v34; // r12
  __int64 v35; // rdx
  unsigned int v36; // esi
  unsigned __int64 v37; // rcx
  __m128i *v38; // r15
  __int64 v39; // r8
  __int64 v40; // rsi
  unsigned __int8 v41; // di
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 i; // rbx
  __int64 v48; // [rsp+0h] [rbp-F0h]
  char v49; // [rsp+8h] [rbp-E8h]
  __int64 v50; // [rsp+8h] [rbp-E8h]
  char v51; // [rsp+8h] [rbp-E8h]
  __int64 v52; // [rsp+10h] [rbp-E0h]
  __int64 v53; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v54; // [rsp+28h] [rbp-C8h]
  __int64 v55; // [rsp+28h] [rbp-C8h]
  __int64 v56; // [rsp+28h] [rbp-C8h]
  __int64 v57; // [rsp+30h] [rbp-C0h]
  unsigned __int8 *v58; // [rsp+38h] [rbp-B8h]
  _DWORD v59[4]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v60; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v61; // [rsp+60h] [rbp-90h] BYREF
  __m128i v62; // [rsp+70h] [rbp-80h] BYREF
  __m128i v63[2]; // [rsp+80h] [rbp-70h] BYREF
  __m128i v64; // [rsp+A0h] [rbp-50h]
  char v65; // [rsp+B0h] [rbp-40h]

  v3 = a2;
  if ( sub_B46500((unsigned __int8 *)a2) )
    return 0;
  v4 = *(_BYTE *)(a2 + 2) & 1;
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    return 0;
  v6 = *(_QWORD *)(a2 - 64);
  v7 = *(_QWORD *)(v6 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 )
    return 0;
  if ( *(_BYTE *)v6 == 91 )
  {
    if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
    {
      v8 = *(_QWORD *)(v6 - 8);
      v9 = *(_BYTE **)v8;
      if ( **(_BYTE **)v8 <= 0x1Cu )
        return v4;
    }
    else
    {
      v8 = v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
      v9 = *(_BYTE **)v8;
      if ( **(_BYTE **)v8 <= 0x1Cu )
        return v4;
    }
    v57 = *(_QWORD *)(v8 + 32);
    if ( v57 )
    {
      v52 = *(_QWORD *)(v8 + 64);
      if ( v52 )
      {
        if ( *v9 == 61 )
        {
          v58 = sub_BD3990(*((unsigned __int8 **)v9 - 4), a2);
          if ( sub_B46500(v9) || (v9[2] & 1) != 0 || *((_QWORD *)v9 + 5) != *(_QWORD *)(a2 + 40) )
            return v4;
          v10 = *((_QWORD *)v9 + 1);
          v11 = *(_QWORD *)(a1 + 184);
          if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
            v10 = **(_QWORD **)(v10 + 16);
          v54 = (sub_9208B0(v11, v10) + 7) & 0xFFFFFFFFFFFFFFF8LL;
          v49 = v12;
          v13.m128i_i64[0] = sub_9208B0(v11, v10);
          v63[0] = v13;
          if ( v13.m128i_i64[0] == v54 && v63[0].m128i_i8[8] == v49 )
          {
            if ( v58 != sub_BD3990(*(unsigned __int8 **)(v3 - 32), v10) )
              return v4;
            sub_2C50240(
              (__int64)v59,
              *(_DWORD *)(v7 + 32),
              v52,
              (__int64)v9,
              *(_QWORD *)(a1 + 176),
              *(_QWORD *)(a1 + 160));
            if ( v59[0] )
            {
              v14 = *(_QWORD **)(a1 + 168);
              sub_D66630(&v60, v3);
              v55 = v3 + 24;
              if ( (_BYTE *)(v3 + 24) != v9 + 24 )
              {
                v50 = v3;
                v15 = v9 + 24;
                v16 = v14;
                v17 = 0;
                while ( 1 )
                {
                  v18 = v15 - 24;
                  if ( !v15 )
                    v18 = 0;
                  v19 = _mm_loadu_si128(&v62);
                  v65 = 1;
                  v20 = _mm_loadu_si128(&v60);
                  v21 = _mm_loadu_si128(&v61);
                  v64 = v19;
                  v63[0] = v20;
                  v63[1] = v21;
                  if ( (sub_CF6520(v16, v18, v63) & 2) != 0 )
                    break;
                  if ( ++v17 > (unsigned int)qword_5010AC8 )
                    break;
                  v15 = (_BYTE *)*((_QWORD *)v15 + 1);
                  if ( (_BYTE *)v55 == v15 )
                  {
                    v3 = v50;
                    goto LABEL_27;
                  }
                }
                v22 = v15;
                v3 = v50;
                if ( (_BYTE *)v55 != v22 )
                  return 0;
              }
LABEL_27:
              v56 = a1 + 200;
              sub_F15FC0(a1 + 200, (__int64)v9);
              if ( v59[0] == 2 )
                sub_2C52B50((__int64)v59, a1 + 8, v52);
              v64.m128i_i16[0] = 257;
              v23 = sub_AD64C0(*(_QWORD *)(v52 + 8), 0, 0);
              v24 = *(_QWORD *)(v3 - 32);
              v60.m128i_i64[0] = v23;
              v25 = *(_QWORD *)(v3 - 64);
              v60.m128i_i64[1] = v52;
              v48 = sub_921130((unsigned int **)(a1 + 8), *(_QWORD *)(v25 + 8), v24, &v60, 2, (__int64)v63, 3u);
              v26 = sub_AA4E30(*(_QWORD *)(a1 + 56));
              v27 = sub_AE5020(v26, *(_QWORD *)(v57 + 8));
              v64.m128i_i16[0] = 257;
              v51 = v27;
              v28 = sub_BD2C40(80, unk_3F10A10);
              v30 = (__int64)v28;
              if ( v28 )
                sub_B4D3C0((__int64)v28, v57, v48, 0, v51, v29, 0, 0);
              (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 96) + 16LL))(
                *(_QWORD *)(a1 + 96),
                v30,
                v63,
                *(_QWORD *)(a1 + 64),
                *(_QWORD *)(a1 + 72));
              v31 = *(_QWORD *)(a1 + 8);
              v32 = 16LL * *(unsigned int *)(a1 + 16);
              if ( v31 != v31 + v32 )
              {
                v53 = v3;
                v33 = v31 + v32;
                v34 = *(_QWORD *)(a1 + 8);
                do
                {
                  v35 = *(_QWORD *)(v34 + 8);
                  v36 = *(_DWORD *)v34;
                  v34 += 16;
                  sub_B99FD0(v30, v36, v35);
                }
                while ( v33 != v34 );
                v3 = v53;
              }
              sub_B47C00(v30, v3, 0, 0);
              LOWORD(v37) = *((_WORD *)v9 + 1);
              v38 = &v60;
              v39 = *(_QWORD *)(a1 + 184);
              v40 = *(_QWORD *)(v57 + 8);
              _BitScanReverse64(&v37, 1LL << ((unsigned __int16)v37 >> 1));
              v41 = 63 - (v37 ^ 0x3F);
              LOWORD(v37) = *(_WORD *)(v3 + 2);
              v63[0].m128i_i8[0] = v41;
              _BitScanReverse64((unsigned __int64 *)&v42, 1LL << ((unsigned __int16)v37 >> 1));
              if ( (unsigned __int8)(63 - (v42 ^ 0x3F)) < v41 )
                v38 = v63;
              v60.m128i_i8[0] = 63 - (v42 ^ 0x3F);
              *(_WORD *)(v30 + 2) = (2 * (unsigned __int8)sub_2C514E0(v38->m128i_i8[0], v40, v52, v39))
                                  | *(_WORD *)(v30 + 2) & 0xFF81;
              sub_BD84D0(v3, v30);
              if ( *(_BYTE *)v30 > 0x1Cu )
              {
                sub_BD6B90((unsigned __int8 *)v30, (unsigned __int8 *)v3);
                for ( i = *(_QWORD *)(v30 + 16); i; i = *(_QWORD *)(i + 8) )
                  sub_F15FC0(v56, *(_QWORD *)(i + 24));
                if ( *(_BYTE *)v30 > 0x1Cu )
                  sub_F15FC0(v56, v30);
              }
              if ( *(_BYTE *)v3 > 0x1Cu )
                sub_F15FC0(v56, v3);
              v4 = 1;
              sub_2C60650(a1, v3, v43, v44, v45, v46);
              return v4;
            }
          }
        }
        return 0;
      }
    }
  }
  return v4;
}
