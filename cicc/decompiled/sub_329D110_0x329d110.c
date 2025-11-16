// Function: sub_329D110
// Address: 0x329d110
//
char __fastcall sub_329D110(__int64 a1, __int64 a2, __int64 a3)
{
  char result; // al
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // r8
  int v15; // r10d
  __int64 v16; // r9
  int v17; // ecx
  __int64 v18; // rcx
  __int64 v19; // r11
  __m128i v20; // [rsp-48h] [rbp-48h]
  __m128i v21; // [rsp-38h] [rbp-38h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v6 = **(_QWORD **)(a1 + 40);
  if ( *(_DWORD *)(a3 + 8) != *(_DWORD *)(v6 + 24) )
    return 0;
  v7 = *(_QWORD *)(a3 + 16);
  v21 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v6 + 40));
  *(_QWORD *)v7 = v21.m128i_i64[0];
  *(_DWORD *)(v7 + 8) = v21.m128i_i32[2];
  v8 = *(_QWORD *)(a3 + 24);
  v20 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v6 + 40) + 40LL));
  *(_QWORD *)v8 = v20.m128i_i64[0];
  *(_DWORD *)(v8 + 8) = v20.m128i_i32[2];
  if ( *(_BYTE *)(a3 + 36) )
  {
    if ( *(_DWORD *)(a3 + 32) != (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v6 + 28)) )
      return 0;
  }
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL);
  if ( *(_DWORD *)(a3 + 40) != *(_DWORD *)(v9 + 24) )
    return 0;
  v10 = *(__int64 **)(v9 + 40);
  v11 = *v10;
  if ( *(_DWORD *)(a3 + 48) != *(_DWORD *)(*v10 + 24) )
    return 0;
  v12 = *(__int64 **)(v11 + 40);
  v13 = *(_QWORD *)(a3 + 56);
  v14 = *v12;
  v15 = *((_DWORD *)v12 + 2);
  v16 = v12[5];
  v17 = *((_DWORD *)v12 + 12);
  if ( v14 != *(_QWORD *)v13
    || v15 != *(_DWORD *)(v13 + 8)
    || (v19 = *(_QWORD *)(a3 + 64), v16 != *(_QWORD *)v19)
    || v17 != *(_DWORD *)(v19 + 8) )
  {
    if ( v16 != *(_QWORD *)v13 )
      return 0;
    if ( v17 != *(_DWORD *)(v13 + 8) )
      return 0;
    v18 = *(_QWORD *)(a3 + 64);
    if ( v14 != *(_QWORD *)v18 || v15 != *(_DWORD *)(v18 + 8) )
      return 0;
  }
  if ( *(_BYTE *)(a3 + 76) && *(_DWORD *)(a3 + 72) != (*(_DWORD *)(a3 + 72) & *(_DWORD *)(v11 + 28)) )
    return 0;
  result = sub_32657E0(a3 + 80, v10[5]);
  if ( !result || *(_BYTE *)(a3 + 100) && *(_DWORD *)(a3 + 96) != (*(_DWORD *)(a3 + 96) & *(_DWORD *)(v9 + 28)) )
    return 0;
  if ( *(_BYTE *)(a3 + 108) )
    return (*(_DWORD *)(a3 + 104) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 104);
  return result;
}
