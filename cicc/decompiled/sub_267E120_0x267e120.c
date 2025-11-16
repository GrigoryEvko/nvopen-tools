// Function: sub_267E120
// Address: 0x267e120
//
__int64 __fastcall sub_267E120(__int64 a1, __int64 a2)
{
  bool v4; // zf
  unsigned __int8 *v5; // rdx
  __int64 v6; // rax
  unsigned int v7; // edi
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 v10; // rax
  unsigned __int8 *v11; // r9
  unsigned int v12; // esi
  __int64 v13; // r11
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 *v16; // rax
  unsigned __int32 i; // edx
  __int64 *v18; // r12
  __int64 v19; // r8
  unsigned __int32 v20; // edx
  __int64 v21; // rax
  __int64 *v22; // r13
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 result; // rax
  int v28; // eax
  __m128i *v29; // r15
  __int64 v30; // rax
  __m128i *v31; // rax
  __m128i *v32; // rbx
  unsigned __int64 v33; // rdi
  int v34; // ebx
  int v35; // ecx
  __m128i v36; // xmm0
  int v37; // r10d
  int v38; // [rsp+Ch] [rbp-84h]
  __int64 v39; // [rsp+18h] [rbp-78h]
  __int64 *v40; // [rsp+28h] [rbp-68h] BYREF
  __m128i v41; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v42[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 (__fastcall *v43)(const __m128i **, const __m128i *, int); // [rsp+50h] [rbp-40h]
  __int64 (__fastcall *v44)(__int64 *, __int64, __int64 *, _BYTE *); // [rsp+58h] [rbp-38h]

  if ( byte_4FF4CE8 )
  {
    v4 = *(_BYTE *)(a1 + 112) == 0;
    *(_QWORD *)(a1 + 104) = 0;
    if ( v4 )
      *(_BYTE *)(a1 + 112) = 1;
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  }
  v5 = sub_250CBE0((__int64 *)(a1 + 72), a2);
  v6 = *(_QWORD *)(a2 + 208);
  v7 = *(_DWORD *)(v6 + 34576);
  v8 = *(_QWORD *)(v6 + 34560);
  if ( v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v10 = v8 + 16LL * v9;
    v11 = *(unsigned __int8 **)v10;
    if ( v5 == *(unsigned __int8 **)v10 )
      goto LABEL_7;
    v28 = 1;
    while ( v11 != (unsigned __int8 *)-4096LL )
    {
      v37 = v28 + 1;
      v9 = (v7 - 1) & (v28 + v9);
      v10 = v8 + 16LL * v9;
      v11 = *(unsigned __int8 **)v10;
      if ( v5 == *(unsigned __int8 **)v10 )
        goto LABEL_7;
      v28 = v37;
    }
  }
  v10 = v8 + 16LL * v7;
LABEL_7:
  *(_DWORD *)(a1 + 120) = *(_DWORD *)(v10 + 8);
  v42[0] = a1;
  v44 = sub_266EF60;
  v42[1] = a2;
  v43 = sub_266E110;
  v41 = (__m128i)(sub_250D070((_QWORD *)(a1 + 72)) & 0xFFFFFFFFFFFFFFFCLL | 1);
  nullsub_1518();
  v12 = *(_DWORD *)(a2 + 56);
  if ( v12 )
  {
    v38 = 1;
    v13 = *(_QWORD *)(a2 + 40);
    v14 = qword_4FEE4D0;
    v39 = qword_4FEE4D8;
    v15 = v41.m128i_i64[0];
    v16 = 0;
    for ( i = (v12 - 1)
            & (((unsigned __int32)v41.m128i_i32[2] >> 9)
             ^ ((unsigned __int32)v41.m128i_i32[2] >> 4)
             ^ (16 * (((unsigned __int32)v41.m128i_i32[0] >> 9) ^ ((unsigned __int32)v41.m128i_i32[0] >> 4))));
          ;
          i = (v12 - 1) & v20 )
    {
      v18 = (__int64 *)(v13 + ((unsigned __int64)i << 6));
      v19 = *v18;
      if ( *(_OWORD *)&v41 == *(_OWORD *)v18 )
      {
        v21 = *((unsigned int *)v18 + 6);
        v22 = v18 + 2;
        v23 = v21;
        if ( (unsigned int)v21 < *((_DWORD *)v18 + 7) )
          goto LABEL_14;
        v29 = (__m128i *)sub_C8D7D0(
                           (__int64)(v18 + 2),
                           (__int64)(v18 + 4),
                           0,
                           0x20u,
                           (unsigned __int64 *)&v40,
                           qword_4FEE4D0);
        v30 = 2LL * *((unsigned int *)v18 + 6);
        v4 = &v29[v30] == 0;
        v31 = &v29[v30];
        v32 = v31;
        if ( !v4 )
        {
          v31[1].m128i_i64[0] = 0;
          if ( v43 )
          {
            v43((const __m128i **)v31, (const __m128i *)v42, 2);
            v32[1].m128i_i64[1] = (__int64)v44;
            v32[1].m128i_i64[0] = (__int64)v43;
          }
        }
        sub_255FA70((__int64)(v18 + 2), v29);
        v33 = v18[2];
        v34 = (int)v40;
        if ( v18 + 4 != (__int64 *)v33 )
          _libc_free(v33);
        ++*((_DWORD *)v18 + 6);
        v18[2] = (__int64)v29;
        *((_DWORD *)v18 + 7) = v34;
        goto LABEL_19;
      }
      if ( qword_4FEE4D0 == v19 && qword_4FEE4D8 == v18[1] )
        break;
      if ( qword_4FEE4C0[0] == v19 && v18[1] == qword_4FEE4C0[1] && !v16 )
        v16 = (__int64 *)(v13 + ((unsigned __int64)i << 6));
      v20 = v38 + i;
      ++v38;
    }
    v35 = *(_DWORD *)(a2 + 48);
    if ( !v16 )
      v16 = (__int64 *)(v13 + ((unsigned __int64)i << 6));
    ++*(_QWORD *)(a2 + 32);
    v15 = (unsigned int)(v35 + 1);
    v40 = v16;
    if ( 4 * (int)v15 >= 3 * v12 )
      goto LABEL_48;
    if ( v12 - *(_DWORD *)(a2 + 52) - (unsigned int)v15 <= v12 >> 3 )
      goto LABEL_49;
  }
  else
  {
    ++*(_QWORD *)(a2 + 32);
    v40 = 0;
LABEL_48:
    v12 *= 2;
LABEL_49:
    sub_2568D00(a2 + 32, v12);
    sub_255C130(a2 + 32, v41.m128i_i64, &v40);
    v14 = qword_4FEE4D0;
    v15 = (unsigned int)(*(_DWORD *)(a2 + 48) + 1);
    v39 = qword_4FEE4D8;
    v16 = v40;
  }
  *(_DWORD *)(a2 + 48) = v15;
  if ( *v16 != v14 || v16[1] != v39 )
    --*(_DWORD *)(a2 + 52);
  v36 = _mm_loadu_si128(&v41);
  v22 = v16 + 2;
  v16[2] = (__int64)(v16 + 4);
  v23 = 0;
  v16[3] = 0x100000000LL;
  *(__m128i *)v16 = v36;
  v21 = 0;
LABEL_14:
  v24 = 32 * v21;
  v4 = *v22 + v24 == 0;
  v25 = *v22 + v24;
  v26 = v25;
  if ( !v4 )
  {
    *(_QWORD *)(v25 + 16) = 0;
    if ( v43 )
    {
      ((void (__fastcall *)(__int64, _QWORD *, __int64, __int64, __int64, __int64))v43)(v25, v42, 2, v15, v19, v14);
      *(_QWORD *)(v26 + 24) = v44;
      *(_QWORD *)(v26 + 16) = v43;
    }
    v23 = *((_DWORD *)v22 + 2);
  }
  *((_DWORD *)v22 + 2) = v23 + 1;
LABEL_19:
  result = (__int64)v43;
  if ( v43 )
    return ((__int64 (__fastcall *)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64))v43)(
             v42,
             v42,
             3,
             v15,
             v19,
             v14);
  return result;
}
