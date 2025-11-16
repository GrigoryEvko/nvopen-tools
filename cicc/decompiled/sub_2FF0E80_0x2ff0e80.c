// Function: sub_2FF0E80
// Address: 0x2ff0e80
//
__int64 __fastcall sub_2FF0E80(__int64 a1, _QWORD *a2, unsigned __int8 a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  char v6; // cl
  __int64 result; // rax
  char v8; // dl
  int v9; // edx
  int v10; // ecx
  unsigned int v11; // r12d
  _BYTE *v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdi
  __m128i *v16; // rax
  __m128i *v17; // rdx
  unsigned __int64 v18; // rcx
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // r14
  __int64 v23; // rsi
  int v24; // eax
  unsigned __int64 v25[2]; // [rsp+20h] [rbp-90h] BYREF
  __m128i v26; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v27; // [rsp+40h] [rbp-70h] BYREF
  size_t v28; // [rsp+48h] [rbp-68h]
  _QWORD v29[2]; // [rsp+50h] [rbp-60h] BYREF
  char *v30; // [rsp+60h] [rbp-50h] BYREF
  size_t v31; // [rsp+68h] [rbp-48h]
  _QWORD v32[8]; // [rsp+70h] [rbp-40h] BYREF

  v4 = a2[2];
  v5 = *(_QWORD *)(a1 + 200);
  if ( *(_QWORD *)(a1 + 184) == v4
    && (v10 = *(_DWORD *)(a1 + 220), *(_DWORD *)(a1 + 220) = v10 + 1, v10 == *(_DWORD *)(a1 + 216)) )
  {
    *(_BYTE *)(a1 + 248) = 1;
    if ( v4 == v5 )
    {
      v24 = *(_DWORD *)(a1 + 236);
      *(_DWORD *)(a1 + 236) = v24 + 1;
      if ( v24 == *(_DWORD *)(a1 + 232) )
        goto LABEL_34;
    }
  }
  else
  {
    v6 = *(_BYTE *)(a1 + 248);
    if ( v4 == v5 )
    {
      v19 = *(_DWORD *)(a1 + 236);
      *(_DWORD *)(a1 + 236) = v19 + 1;
      if ( v19 == *(_DWORD *)(a1 + 232) )
      {
LABEL_34:
        *(_BYTE *)(a1 + 249) = 1;
        goto LABEL_4;
      }
    }
    if ( !v6 )
    {
LABEL_4:
      (*(void (__fastcall **)(_QWORD *))(*a2 + 8LL))(a2);
      goto LABEL_5;
    }
  }
  if ( *(_BYTE *)(a1 + 249) )
    goto LABEL_4;
  v11 = a3;
  if ( !*(_BYTE *)(a1 + 250) )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD))(**(_QWORD **)(a1 + 176) + 16LL))(*(_QWORD *)(a1 + 176), a2, a3);
    goto LABEL_36;
  }
  v12 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD *))(*a2 + 16LL))(a2);
  v30 = (char *)v32;
  sub_2FEE630((__int64 *)&v30, v12, (__int64)&v12[v13]);
  v27 = v29;
  sub_2FEE630((__int64 *)&v27, "After ", (__int64)"");
  v14 = 15;
  v15 = 15;
  if ( v27 != (_BYTE *)v29 )
    v15 = v29[0];
  if ( v28 + v31 > v15 )
  {
    if ( v30 != (char *)v32 )
      v14 = v32[0];
    if ( v28 + v31 <= v14 )
    {
      v16 = (__m128i *)sub_2241130((unsigned __int64 *)&v30, 0, 0, v27, v28);
      v17 = v16 + 1;
      v25[0] = (unsigned __int64)&v26;
      v18 = v16->m128i_i64[0];
      if ( (__m128i *)v16->m128i_i64[0] != &v16[1] )
        goto LABEL_24;
LABEL_49:
      v26 = _mm_loadu_si128(v16 + 1);
      goto LABEL_25;
    }
  }
  v16 = (__m128i *)sub_2241490((unsigned __int64 *)&v27, v30, v31);
  v17 = v16 + 1;
  v25[0] = (unsigned __int64)&v26;
  v18 = v16->m128i_i64[0];
  if ( (__m128i *)v16->m128i_i64[0] == &v16[1] )
    goto LABEL_49;
LABEL_24:
  v25[0] = v18;
  v26.m128i_i64[0] = v16[1].m128i_i64[0];
LABEL_25:
  v25[1] = v16->m128i_u64[1];
  v16->m128i_i64[0] = (__int64)v17;
  v16->m128i_i64[1] = 0;
  v16[1].m128i_i8[0] = 0;
  if ( v27 != (_BYTE *)v29 )
    j_j___libc_free_0((unsigned __int64)v27);
  if ( v30 != (char *)v32 )
    j_j___libc_free_0((unsigned __int64)v30);
  sub_2FF0DD0(a1, 1);
  (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD))(**(_QWORD **)(a1 + 176) + 16LL))(*(_QWORD *)(a1 + 176), a2, v11);
  sub_2FF0E00(a1, (__int64)v25);
  if ( (__m128i *)v25[0] != &v26 )
    j_j___libc_free_0(v25[0]);
LABEL_36:
  v20 = *(_QWORD *)(a1 + 264);
  v21 = *(_QWORD *)(v20 + 32);
  v22 = v21 + 24LL * *(unsigned int *)(v20 + 40);
  if ( v21 != v22 )
  {
    if ( *(_QWORD *)v21 == v4 )
      goto LABEL_40;
    while ( 1 )
    {
      v21 += 24;
      if ( v22 == v21 )
        break;
      if ( *(_QWORD *)v21 == v4 )
      {
LABEL_40:
        v23 = *(_QWORD *)(v21 + 8);
        if ( !*(_BYTE *)(v21 + 16) )
          v23 = sub_BB95C0(*(_QWORD *)(v21 + 8), v23);
        sub_2FF0E80(a1, v23, v11);
      }
    }
  }
LABEL_5:
  result = *(_QWORD *)(a1 + 192);
  if ( *(_QWORD *)(a1 + 208) == v4 )
  {
    v9 = *(_DWORD *)(a1 + 244);
    *(_DWORD *)(a1 + 244) = v9 + 1;
    if ( v9 == *(_DWORD *)(a1 + 240) )
    {
      *(_BYTE *)(a1 + 249) = 1;
      if ( v4 == result )
      {
        result = *(unsigned int *)(a1 + 228);
        *(_DWORD *)(a1 + 228) = result + 1;
        if ( *(_DWORD *)(a1 + 224) == (_DWORD)result )
          goto LABEL_32;
      }
LABEL_11:
      if ( !*(_BYTE *)(a1 + 248) )
        sub_C64ED0("Cannot stop compilation after pass that is not run", 1u);
      return result;
    }
  }
  v8 = *(_BYTE *)(a1 + 249);
  if ( v4 == result )
  {
    result = *(unsigned int *)(a1 + 228);
    *(_DWORD *)(a1 + 228) = result + 1;
    if ( (_DWORD)result == *(_DWORD *)(a1 + 224) )
    {
LABEL_32:
      *(_BYTE *)(a1 + 248) = 1;
      return result;
    }
  }
  if ( v8 )
    goto LABEL_11;
  return result;
}
