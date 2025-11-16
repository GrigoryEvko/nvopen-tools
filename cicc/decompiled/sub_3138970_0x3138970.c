// Function: sub_3138970
// Address: 0x3138970
//
__int64 __fastcall sub_3138970(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        char a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v8; // r15
  __int64 v10; // r13
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rdi
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rdx
  unsigned int v20; // esi
  char v21; // al
  __int64 v22; // rdx
  __int16 v24; // dx
  __int64 v25; // rsi
  char v26; // al
  __int64 v27; // rax
  __m128i *v28; // roff
  __m128i v29; // xmm0
  void (__fastcall *v30)(__m128i *, __m128i *, __int64); // rdx
  void (__fastcall *v31)(__int64 *, __m128i *, __m128i *); // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdi
  void (__fastcall *v35)(__int64, __int64, __int64); // rax
  __int64 v36; // rcx
  unsigned __int64 v37; // rsi
  unsigned __int64 v38; // rax
  int v39; // edx
  unsigned __int64 v40; // rax
  void (__fastcall *v41)(__m128i *, __m128i *, __int64, __int64, __int64); // rax
  __int64 v42; // [rsp+8h] [rbp-98h]
  __int64 v43; // [rsp+8h] [rbp-98h]
  __int64 v44; // [rsp+8h] [rbp-98h]
  __int64 v45; // [rsp+18h] [rbp-88h] BYREF
  __m128i v46; // [rsp+20h] [rbp-80h] BYREF
  __int64 v47; // [rsp+30h] [rbp-70h]
  __m128i v48; // [rsp+40h] [rbp-60h] BYREF
  void (__fastcall *v49)(__m128i *, __m128i *, __int64); // [rsp+50h] [rbp-50h]
  void (__fastcall *v50)(__int64 *, __m128i *, __m128i *); // [rsp+58h] [rbp-48h]
  __int32 v51; // [rsp+60h] [rbp-40h]
  __int8 v52; // [rsp+64h] [rbp-3Ch]

  v8 = a2 + 512;
  v10 = a2;
  v13 = a7;
  if ( (_QWORD)a7 )
  {
    a2 = a7;
    v42 = a7;
    sub_A88F30(v8, a7, *((__int64 *)&a7 + 1), a8);
    v13 = v42;
    if ( !a5 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)(a2 + 560) = 0;
    *(_QWORD *)(a2 + 568) = 0;
    *(_WORD *)(a2 + 576) = 0;
    if ( !a5 )
    {
LABEL_3:
      if ( a4 )
      {
        sub_B43D10(a4);
        v14 = *(_QWORD *)(v8 + 56);
        v15 = *(_QWORD *)(v8 + 64);
        v16 = *(_QWORD *)(v10 + 600);
        LOWORD(v51) = 257;
        (*(void (__fastcall **)(__int64, _QWORD *, __m128i *, __int64, __int64))(*(_QWORD *)v16 + 16LL))(
          v16,
          a4,
          &v48,
          v14,
          v15);
        v17 = *(_QWORD *)(v10 + 512);
        v18 = v17 + 16LL * *(unsigned int *)(v10 + 520);
        while ( v18 != v17 )
        {
          v19 = *(_QWORD *)(v17 + 8);
          v20 = *(_DWORD *)v17;
          v17 += 16;
          sub_B99FD0((__int64)a4, v20, v19);
        }
        v21 = *(_BYTE *)(a1 + 24);
        v22 = a4[5];
        *(_QWORD *)(a1 + 8) = a4 + 3;
        *(_QWORD *)a1 = v22;
        *(_BYTE *)(a1 + 24) = v21 & 0xFC | 2;
        *(_WORD *)(a1 + 16) = 0;
      }
      else
      {
        v24 = *(_WORD *)(v10 + 576);
        v25 = *(_QWORD *)(v10 + 560);
        v26 = *(_BYTE *)(a1 + 24) & 0xFC;
        *(_QWORD *)(a1 + 8) = *(_QWORD *)(v10 + 568);
        *(_QWORD *)a1 = v25;
        *(_BYTE *)(a1 + 24) = v26 | 2;
        *(_WORD *)(a1 + 16) = v24;
      }
      return a1;
    }
  }
  v27 = *(unsigned int *)(v10 + 8);
  v49 = 0;
  v28 = (__m128i *)(*(_QWORD *)v10 + 40 * v27 - 40);
  v29 = _mm_loadu_si128(v28);
  *v28 = _mm_loadu_si128(&v48);
  v48 = v29;
  v30 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v28[1].m128i_i64[0];
  v28[1].m128i_i64[0] = 0;
  v49 = v30;
  v31 = (void (__fastcall *)(__int64 *, __m128i *, __m128i *))v28[1].m128i_i64[1];
  v28[1].m128i_i64[1] = (__int64)v50;
  v50 = v31;
  v51 = v28[2].m128i_i32[0];
  v52 = v28[2].m128i_i8[4];
  v32 = (unsigned int)(*(_DWORD *)(v10 + 8) - 1);
  *(_DWORD *)(v10 + 8) = v32;
  v33 = 5 * v32;
  v34 = *(_QWORD *)v10 + 40 * v32;
  v35 = *(void (__fastcall **)(__int64, __int64, __int64))(v34 + 16);
  if ( v35 )
  {
    v43 = v13;
    a2 = v34;
    v35(v34, v34, 3);
    v13 = v43;
  }
  v47 = a8;
  v46 = _mm_loadu_si128((const __m128i *)&a7);
  if ( !v49 )
    sub_4263D6(v34, a2, v33);
  v44 = v13;
  v50(&v45, &v48, &v46);
  v37 = v45 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v45 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    v38 = *(_QWORD *)(v44 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v38 == v44 + 48 )
    {
      v37 = 0;
    }
    else
    {
      if ( !v38 )
        BUG();
      v39 = *(unsigned __int8 *)(v38 - 24);
      v40 = v38 - 24;
      if ( (unsigned int)(v39 - 30) < 0xB )
        v37 = v40;
    }
    sub_D5F1F0(v8, v37);
    if ( v49 )
      v49(&v48, &v48, 3);
    goto LABEL_3;
  }
  v41 = (void (__fastcall *)(__m128i *, __m128i *, __int64, __int64, __int64))v49;
  *(_BYTE *)(a1 + 24) |= 3u;
  *(_QWORD *)a1 = v37;
  if ( v41 )
    v41(&v48, &v48, 3, v36, v44);
  return a1;
}
