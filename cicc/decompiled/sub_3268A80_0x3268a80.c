// Function: sub_3268A80
// Address: 0x3268a80
//
void __fastcall sub_3268A80(_QWORD *a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  bool v7; // zf
  __int64 v8; // rcx
  __int64 v9; // rax
  unsigned int v10; // ebx
  __m128i si128; // xmm1
  const void *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // r15
  char *v17; // rax
  char v18; // dl
  int v19; // eax
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // r12
  unsigned int v27; // edx
  __int64 v28; // r15
  _OWORD **v29; // rdi
  __int64 v30; // r15
  __int64 v31; // r12
  __m128i v32; // xmm0
  unsigned __int128 v33; // [rsp+0h] [rbp-1A0h] BYREF
  __m128i v34; // [rsp+10h] [rbp-190h] BYREF
  bool v35; // [rsp+27h] [rbp-179h]
  _OWORD **v36; // [rsp+28h] [rbp-178h]
  __int64 v37; // [rsp+30h] [rbp-170h]
  _OWORD *v38; // [rsp+38h] [rbp-168h]
  _OWORD *v39; // [rsp+40h] [rbp-160h] BYREF
  __int64 v40; // [rsp+48h] [rbp-158h]
  _OWORD v41[8]; // [rsp+50h] [rbp-150h] BYREF
  __int64 v42; // [rsp+D0h] [rbp-D0h] BYREF
  char *v43; // [rsp+D8h] [rbp-C8h]
  __int64 v44; // [rsp+E0h] [rbp-C0h]
  int v45; // [rsp+E8h] [rbp-B8h]
  unsigned __int8 v46; // [rsp+ECh] [rbp-B4h]
  char v47; // [rsp+F0h] [rbp-B0h] BYREF

  v38 = v41;
  v7 = *(_DWORD *)(a2 + 24) == 298;
  v39 = v41;
  v40 = 0x800000000LL;
  v37 = a2;
  v33 = __PAIR128__(a4, a3);
  v42 = 0;
  v43 = &v47;
  v44 = 16;
  v45 = 0;
  v46 = 1;
  v35 = 0;
  if ( v7 && (*(_BYTE *)(*(_QWORD *)(a2 + 112) + 37LL) & 0xF) == 0 )
    v35 = (*(_BYTE *)(a2 + 32) & 8) == 0;
  v8 = 1;
  LODWORD(v9) = 1;
  v10 = 0;
  si128 = _mm_load_si128((const __m128i *)&v33);
  v13 = v38;
  LODWORD(v40) = 1;
  v41[0] = si128;
  while ( 1 )
  {
    v14 = (__int64)v13 + 16 * (unsigned int)v9 - 16;
    v15 = *(_QWORD *)v14;
    v16 = *(unsigned int *)(v14 + 8);
    LODWORD(v40) = v9 - 1;
    if ( (_BYTE)v8 )
    {
      v17 = v43;
      v14 = (__int64)&v43[8 * HIDWORD(v44)];
      if ( v43 != (char *)v14 )
      {
        while ( v15 != *(_QWORD *)v17 )
        {
          v17 += 8;
          if ( (char *)v14 == v17 )
            goto LABEL_22;
        }
LABEL_8:
        LODWORD(v9) = v40;
        goto LABEL_9;
      }
LABEL_22:
      if ( HIDWORD(v44) < (unsigned int)v44 )
        break;
    }
    sub_C8CC70((__int64)&v42, v15, v14, v8, a5, a6);
    v8 = v46;
    if ( !v18 )
      goto LABEL_8;
    if ( *(_DWORD *)(a1[1] + 536964LL) < v10 )
      goto LABEL_24;
LABEL_13:
    v19 = *(_DWORD *)(v15 + 24);
    if ( v19 == 2 )
    {
      v20 = *(_DWORD *)(v15 + 64);
      if ( v20 <= 0x10 )
      {
        v9 = (unsigned int)v40;
        if ( v20 )
        {
          v29 = &v39;
          v30 = v15;
          a6 = 40LL * (v20 - 1);
          v31 = a6;
          do
          {
            v32 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v30 + 40) + v31));
            if ( v9 + 1 > (unsigned __int64)HIDWORD(v40) )
            {
              v36 = v29;
              v34 = v32;
              sub_C8D5F0((__int64)v29, v38, v9 + 1, 0x10u, a5, a6);
              v9 = (unsigned int)v40;
              v32 = _mm_load_si128(&v34);
              v29 = v36;
            }
            v31 -= 40;
            v39[v9] = v32;
            v9 = (unsigned int)(v40 + 1);
            LODWORD(v40) = v40 + 1;
          }
          while ( v31 != -40 );
          goto LABEL_36;
        }
        goto LABEL_37;
      }
    }
    else
    {
      if ( v19 > 299 )
      {
        if ( (unsigned int)(v19 - 366) > 1 )
          goto LABEL_15;
        goto LABEL_39;
      }
      if ( v19 > 297 )
      {
        if ( v19 == 298
          && (*(_BYTE *)(*(_QWORD *)(v15 + 112) + 37LL) & 0xF) == 0
          && (*(_BYTE *)(v15 + 32) & 8) == 0
          && v35 )
        {
LABEL_32:
          v25 = *(__int64 **)(v15 + 40);
          v26 = *v25;
          v27 = *((_DWORD *)v25 + 2);
          v9 = (unsigned int)v40;
          if ( v26 )
          {
            v28 = v27;
            if ( (unsigned __int64)(unsigned int)v40 + 1 > HIDWORD(v40) )
            {
              sub_C8D5F0((__int64)&v39, v38, (unsigned int)v40 + 1LL, 0x10u, a5, a6);
              v9 = (unsigned int)v40;
            }
            v9 = (__int64)&v39[v9];
            *(_QWORD *)v9 = v26;
            *(_QWORD *)(v9 + 8) = v28;
            LODWORD(v9) = v40 + 1;
            LODWORD(v40) = v40 + 1;
          }
LABEL_36:
          v8 = v46;
LABEL_37:
          ++v10;
LABEL_9:
          if ( !(_DWORD)v9 )
            goto LABEL_18;
          goto LABEL_10;
        }
LABEL_39:
        if ( !sub_3267610(a1, v37, v15) )
          goto LABEL_32;
        goto LABEL_15;
      }
      if ( v19 == 1 )
      {
        LODWORD(v9) = v40;
        goto LABEL_37;
      }
      if ( v19 == 50 )
        goto LABEL_32;
    }
LABEL_15:
    v21 = *(unsigned int *)(a5 + 8);
    if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
    {
      sub_C8D5F0(a5, (const void *)(a5 + 16), v21 + 1, 0x10u, a5, a6);
      v21 = *(unsigned int *)(a5 + 8);
    }
    v9 = *(_QWORD *)a5 + 16 * v21;
    *(_QWORD *)v9 = v15;
    *(_QWORD *)(v9 + 8) = v16;
    LODWORD(v9) = v40;
    ++*(_DWORD *)(a5 + 8);
    v8 = v46;
    if ( !(_DWORD)v9 )
    {
LABEL_18:
      if ( (_BYTE)v8 )
        goto LABEL_19;
      goto LABEL_27;
    }
LABEL_10:
    v13 = v39;
  }
  ++HIDWORD(v44);
  *(_QWORD *)v14 = v15;
  v22 = a1[1];
  ++v42;
  v8 = v46;
  if ( *(_DWORD *)(v22 + 536964) >= v10 )
    goto LABEL_13;
LABEL_24:
  v23 = *(_DWORD *)(a5 + 12);
  *(_DWORD *)(a5 + 8) = 0;
  v24 = 0;
  if ( !v23 )
  {
    sub_C8D5F0(a5, (const void *)(a5 + 16), 1u, 0x10u, a5, a6);
    v24 = 16LL * *(unsigned int *)(a5 + 8);
  }
  *(__m128i *)(*(_QWORD *)a5 + v24) = _mm_load_si128((const __m128i *)&v33);
  ++*(_DWORD *)(a5 + 8);
  if ( !v46 )
LABEL_27:
    _libc_free((unsigned __int64)v43);
LABEL_19:
  if ( v39 != v38 )
    _libc_free((unsigned __int64)v39);
}
