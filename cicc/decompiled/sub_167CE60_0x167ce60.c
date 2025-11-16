// Function: sub_167CE60
// Address: 0x167ce60
//
__int64 __fastcall sub_167CE60(__m128i *a1, __int64 a2, __int64 a3)
{
  void (__fastcall *v4)(__m128i *, __int64); // rax
  __int64 v5; // rsi
  __m128i v6; // xmm0
  __int64 (__fastcall *v7)(_QWORD, _QWORD, _QWORD); // rcx
  __int64 v8; // r12
  __int64 result; // rax
  size_t v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r9
  unsigned int v13; // r13d
  unsigned int v14; // r8d
  __int64 v15; // rcx
  __int64 *v16; // rbx
  __int64 *v17; // r13
  size_t v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // eax
  int v23; // edi
  __int64 v24; // r9
  int v25; // ecx
  __int64 *v26; // rdx
  __int64 v27; // rsi
  int v28; // r15d
  int v29; // eax
  int v30; // eax
  int v31; // esi
  __int64 v32; // r9
  int v33; // r8d
  __int64 *v34; // rdi
  unsigned int v35; // r13d
  __int64 *v36; // r11
  int v37; // r10d
  __int64 *v38; // r8
  __int64 v39; // [rsp+8h] [rbp-68h]
  char v40; // [rsp+1Fh] [rbp-51h] BYREF
  __m128i v41; // [rsp+20h] [rbp-50h] BYREF
  __int64 (__fastcall *v42)(_QWORD, _QWORD, _QWORD); // [rsp+30h] [rbp-40h]
  void (__fastcall *v43)(__m128i *, __int64); // [rsp+38h] [rbp-38h]

  v4 = *(void (__fastcall **)(__m128i *, __int64))(a3 + 24);
  v5 = (__int64)v43;
  v6 = _mm_loadu_si128((const __m128i *)a3);
  v7 = *(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(a3 + 16);
  *(__m128i *)a3 = _mm_loadu_si128(&v41);
  *(_QWORD *)(a3 + 16) = 0;
  *(_QWORD *)(a3 + 24) = v5;
  v8 = a1->m128i_i64[0];
  v43 = v4;
  LOBYTE(v4) = *(_BYTE *)(a2 + 32);
  v42 = v7;
  v41 = v6;
  result = ((_BYTE)v4 + 15) & 0xF;
  if ( (unsigned __int8)result > 2u && (*(_BYTE *)(v8 + 72) & 2) == 0 )
    goto LABEL_25;
  if ( *(_QWORD *)(v8 + 128) )
  {
    a1 = (__m128i *)(v8 + 80);
    v5 = (__int64)sub_1649960(a2);
    sub_167C570(v8 + 80, (const void *)v5, v10);
    v7 = v42;
  }
  if ( !v7 )
LABEL_57:
    sub_4263D6(a1, v5, a3);
  v43(&v41, a2);
  result = sub_15E4F10(a2);
  v11 = result;
  if ( !result )
    goto LABEL_24;
  v5 = *(unsigned int *)(v8 + 216);
  a1 = (__m128i *)(v8 + 192);
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(v8 + 192);
    goto LABEL_31;
  }
  v12 = *(_QWORD *)(v8 + 200);
  v13 = ((unsigned int)result >> 9) ^ ((unsigned int)result >> 4);
  v14 = (v5 - 1) & v13;
  result = v12 + 32LL * v14;
  v15 = *(_QWORD *)result;
  if ( v11 == *(_QWORD *)result )
  {
    v16 = *(__int64 **)(result + 8);
    v17 = *(__int64 **)(result + 16);
    goto LABEL_9;
  }
  v28 = 1;
  v26 = 0;
  while ( 1 )
  {
    if ( v15 == -8 )
    {
      if ( !v26 )
        v26 = (__int64 *)result;
      v29 = *(_DWORD *)(v8 + 208);
      ++*(_QWORD *)(v8 + 192);
      v25 = v29 + 1;
      if ( 4 * (v29 + 1) < (unsigned int)(3 * v5) )
      {
        result = (unsigned int)(v5 - *(_DWORD *)(v8 + 212) - v25);
        if ( (unsigned int)result > (unsigned int)v5 >> 3 )
        {
LABEL_33:
          *(_DWORD *)(v8 + 208) = v25;
          if ( *v26 != -8 )
            --*(_DWORD *)(v8 + 212);
          *v26 = v11;
          v26[1] = 0;
          v26[2] = 0;
          v26[3] = 0;
          goto LABEL_24;
        }
        sub_167C9F0((__int64)a1, v5);
        v30 = *(_DWORD *)(v8 + 216);
        if ( v30 )
        {
          v31 = v30 - 1;
          v32 = *(_QWORD *)(v8 + 200);
          v33 = 1;
          v34 = 0;
          v35 = (v30 - 1) & v13;
          v25 = *(_DWORD *)(v8 + 208) + 1;
          v26 = (__int64 *)(v32 + 32LL * v35);
          result = *v26;
          if ( v11 != *v26 )
          {
            while ( result != -8 )
            {
              if ( result == -16 && !v34 )
                v34 = v26;
              v35 = v31 & (v33 + v35);
              v26 = (__int64 *)(v32 + 32LL * v35);
              result = *v26;
              if ( v11 == *v26 )
                goto LABEL_33;
              ++v33;
            }
            if ( v34 )
              v26 = v34;
          }
          goto LABEL_33;
        }
LABEL_70:
        ++*(_DWORD *)(v8 + 208);
        BUG();
      }
LABEL_31:
      sub_167C9F0((__int64)a1, 2 * v5);
      v22 = *(_DWORD *)(v8 + 216);
      if ( v22 )
      {
        v23 = v22 - 1;
        v24 = *(_QWORD *)(v8 + 200);
        result = (v22 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v25 = *(_DWORD *)(v8 + 208) + 1;
        v26 = (__int64 *)(v24 + 32 * result);
        v27 = *v26;
        if ( v11 != *v26 )
        {
          v37 = 1;
          v38 = 0;
          while ( v27 != -8 )
          {
            if ( v27 == -16 && !v38 )
              v38 = v26;
            result = v23 & (unsigned int)(v37 + result);
            v26 = (__int64 *)(v24 + 32LL * (unsigned int)result);
            v27 = *v26;
            if ( v11 == *v26 )
              goto LABEL_33;
            ++v37;
          }
          if ( v38 )
            v26 = v38;
        }
        goto LABEL_33;
      }
      goto LABEL_70;
    }
    if ( v15 != -16 || v26 )
      result = (__int64)v26;
    a3 = (unsigned int)(v28 + 1);
    v14 = (v5 - 1) & (v28 + v14);
    v36 = (__int64 *)(v12 + 32LL * v14);
    v15 = *v36;
    if ( v11 == *v36 )
      break;
    ++v28;
    v26 = (__int64 *)result;
    result = v12 + 32LL * v14;
  }
  v16 = (__int64 *)v36[1];
  v17 = (__int64 *)v36[2];
LABEL_9:
  while ( v17 != v16 )
  {
LABEL_16:
    v19 = *v16;
    if ( (*(_BYTE *)(*v16 + 23) & 0x20) != 0
      && ((*(_BYTE *)(v19 + 32) + 9) & 0xFu) > 1
      && (v39 = **(_QWORD **)v8,
          v5 = (__int64)sub_1649960(*v16),
          a1 = (__m128i *)v39,
          v21 = sub_1632000(v39, v5, v20),
          (a3 = v21) != 0)
      && (*(_BYTE *)(v21 + 32) & 0xFu) - 7 > 1 )
    {
      v40 = 1;
      if ( (*(_BYTE *)(v8 + 72) & 1) == 0 )
      {
        v5 = (__int64)&v40;
        a1 = (__m128i *)v8;
        result = sub_167C240(v8, (bool *)&v40, v21, v19);
        if ( (_BYTE)result )
          break;
        if ( !v40 )
        {
          if ( v17 == ++v16 )
            break;
          goto LABEL_16;
        }
      }
    }
    else
    {
      v40 = 1;
    }
    if ( *(_QWORD *)(v8 + 128) )
    {
      a1 = (__m128i *)(v8 + 80);
      v5 = (__int64)sub_1649960(v19);
      sub_167C570(v8 + 80, (const void *)v5, v18);
    }
    if ( !v42 )
      goto LABEL_57;
    ++v16;
    v5 = v19;
    a1 = &v41;
    result = ((__int64 (__fastcall *)(__m128i *, __int64))v43)(&v41, v19);
  }
LABEL_24:
  v7 = v42;
LABEL_25:
  if ( v7 )
    return v7(&v41, &v41, 3);
  return result;
}
