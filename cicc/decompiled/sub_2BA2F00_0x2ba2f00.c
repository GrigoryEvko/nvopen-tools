// Function: sub_2BA2F00
// Address: 0x2ba2f00
//
__int64 __fastcall sub_2BA2F00(__int64 a1, const __m128i *a2)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned int v11; // eax
  __int64 v12; // rbx
  int v13; // eax
  unsigned int v14; // esi
  unsigned int v15; // ecx
  __int64 v16; // rdx
  __m128i v17; // xmm0
  unsigned __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  int v21; // eax
  __int64 v22; // rcx
  __m128i *v23; // rsi
  __int64 v24; // rdx
  _QWORD *v25; // rdi
  __int64 v26; // r12
  unsigned __int64 *v27; // r15
  unsigned __int64 *v28; // r12
  unsigned __int64 v29; // r12
  __int64 v30; // rdi
  __int64 v31; // [rsp+0h] [rbp-110h] BYREF
  __int64 v32; // [rsp+8h] [rbp-108h]
  __int64 v33; // [rsp+10h] [rbp-100h]
  int v34; // [rsp+18h] [rbp-F8h]
  _QWORD v35[2]; // [rsp+20h] [rbp-F0h] BYREF
  char v36; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v37; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v38; // [rsp+80h] [rbp-90h]
  unsigned __int64 *v39; // [rsp+88h] [rbp-88h]
  __int64 v40; // [rsp+90h] [rbp-80h]
  _BYTE v41[120]; // [rsp+98h] [rbp-78h] BYREF

  v4 = a2->m128i_i64[1];
  v5 = a2->m128i_i64[0];
  v34 = 0;
  v6 = a2[1].m128i_i64[0];
  v32 = v4;
  v31 = v5;
  v33 = v6;
  if ( (unsigned __int8)sub_2B47D60(a1, &v31, v35) )
  {
    v9 = *(unsigned int *)(v35[0] + 24LL);
    return *(_QWORD *)(a1 + 272) + 104 * v9 + 24;
  }
  v11 = *(_DWORD *)(a1 + 8);
  v12 = v35[0];
  ++*(_QWORD *)a1;
  v37.m128i_i64[0] = v12;
  v13 = (v11 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v15 = 24;
    v14 = 8;
  }
  else
  {
    v14 = *(_DWORD *)(a1 + 24);
    v15 = 3 * v14;
  }
  if ( v15 <= 4 * v13 )
  {
    sub_2BA2A80(a1, 2 * v14);
  }
  else
  {
    if ( v14 - (v13 + *(_DWORD *)(a1 + 12)) > v14 >> 3 )
      goto LABEL_8;
    sub_2BA2A80(a1, v14);
  }
  sub_2B47D60(a1, &v31, &v37);
  v12 = v37.m128i_i64[0];
  v13 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
LABEL_8:
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v13);
  if ( *(_QWORD *)(v12 + 16) != -4096 || *(_QWORD *)(v12 + 8) != -4096 || *(_QWORD *)v12 != -4096 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)(v12 + 16) = v33;
  *(_QWORD *)(v12 + 8) = v32;
  *(_QWORD *)v12 = v31;
  *(_DWORD *)(v12 + 24) = v34;
  v16 = a2[1].m128i_i64[0];
  v17 = _mm_loadu_si128(a2);
  v18 = *(unsigned int *)(a1 + 284);
  v35[0] = &v36;
  v38 = v16;
  v19 = *(unsigned int *)(a1 + 280);
  v35[1] = 0x100000000LL;
  v20 = v19 + 1;
  v40 = 0x100000000LL;
  v21 = v19;
  v39 = (unsigned __int64 *)v41;
  v37 = v17;
  if ( v19 + 1 > v18 )
  {
    v29 = *(_QWORD *)(a1 + 272);
    v30 = a1 + 272;
    if ( v29 > (unsigned __int64)&v37 || (unsigned __int64)&v37 >= v29 + 104 * v19 )
    {
      sub_2B55EB0(v30, v20, v19, v18, v7, v8);
      v19 = *(unsigned int *)(a1 + 280);
      v22 = *(_QWORD *)(a1 + 272);
      v23 = &v37;
      v21 = *(_DWORD *)(a1 + 280);
    }
    else
    {
      sub_2B55EB0(v30, v20, v19, v18, v7, v8);
      v22 = *(_QWORD *)(a1 + 272);
      v19 = *(unsigned int *)(a1 + 280);
      v23 = (__m128i *)((char *)&v37 + v22 - v29);
      v21 = *(_DWORD *)(a1 + 280);
    }
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 272);
    v23 = &v37;
  }
  v24 = 13 * v19;
  v25 = (_QWORD *)(v22 + 8 * v24);
  if ( v25 )
  {
    *v25 = v23->m128i_i64[0];
    v25[1] = v23->m128i_i64[1];
    v25[2] = v23[1].m128i_i64[0];
    v25[3] = v25 + 5;
    v25[4] = 0x100000000LL;
    if ( v23[2].m128i_i32[0] )
      sub_2B42980((__int64)(v25 + 3), (__int64)&v23[1].m128i_i64[1], v24, v22, v7, v8);
    v21 = *(_DWORD *)(a1 + 280);
  }
  v26 = (unsigned int)v40;
  v27 = v39;
  *(_DWORD *)(a1 + 280) = v21 + 1;
  v28 = &v27[8 * v26];
  if ( v27 != v28 )
  {
    do
    {
      v28 -= 8;
      if ( (unsigned __int64 *)*v28 != v28 + 2 )
        _libc_free(*v28);
    }
    while ( v27 != v28 );
    v28 = v39;
  }
  if ( v28 != (unsigned __int64 *)v41 )
    _libc_free((unsigned __int64)v28);
  v9 = (unsigned int)(*(_DWORD *)(a1 + 280) - 1);
  *(_DWORD *)(v12 + 24) = v9;
  return *(_QWORD *)(a1 + 272) + 104 * v9 + 24;
}
