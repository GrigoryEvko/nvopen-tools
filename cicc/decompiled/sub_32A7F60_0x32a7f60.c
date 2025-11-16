// Function: sub_32A7F60
// Address: 0x32a7f60
//
bool __fastcall sub_32A7F60(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  _QWORD *v6; // rax
  int v7; // edx
  __int64 v8; // r13
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 *v11; // r8
  __int64 v12; // rdi
  int v13; // eax
  char v14; // r14
  _QWORD *v15; // rdx
  __int64 v16; // rax
  unsigned __int64 *v17; // r8
  __int64 v18; // rdi
  int v19; // eax
  char v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  __m128i v23; // [rsp-58h] [rbp-58h]
  __m128i v24; // [rsp-48h] [rbp-48h]
  unsigned __int64 v25; // [rsp-38h] [rbp-38h] BYREF
  unsigned int v26; // [rsp-30h] [rbp-30h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v6 = *(_QWORD **)(a1 + 40);
  v7 = *(_DWORD *)(a3 + 8);
  v8 = *v6;
  if ( v7 != *(_DWORD *)(*v6 + 24LL) )
    goto LABEL_4;
  v10 = *(_QWORD *)(a3 + 16);
  v24 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v8 + 40));
  *(_QWORD *)v10 = v24.m128i_i64[0];
  *(_DWORD *)(v10 + 8) = v24.m128i_i32[2];
  v11 = *(unsigned __int64 **)(a3 + 24);
  v12 = *(_QWORD *)(*(_QWORD *)(v8 + 40) + 40LL);
  if ( v12 )
  {
    v13 = *(_DWORD *)(v12 + 24);
    if ( v13 == 11 || v13 == 35 )
    {
      if ( v11 )
      {
        v21 = *(_QWORD *)(v12 + 96);
        if ( *((_DWORD *)v11 + 2) > 0x40u || *(_DWORD *)(v21 + 32) > 0x40u )
        {
          sub_C43990(*(_QWORD *)(a3 + 24), v21 + 24);
          v15 = *(_QWORD **)(a1 + 40);
          goto LABEL_33;
        }
        *v11 = *(_QWORD *)(v21 + 24);
        *((_DWORD *)v11 + 2) = *(_DWORD *)(v21 + 32);
      }
      v15 = *(_QWORD **)(a1 + 40);
LABEL_33:
      v6 = v15;
      if ( !*(_BYTE *)(a3 + 36) || *(_DWORD *)(a3 + 32) == (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v8 + 28)) )
      {
        if ( (unsigned __int8)sub_32657E0(a3 + 40, v15[5]) )
          goto LABEL_29;
        v6 = *(_QWORD **)(a1 + 40);
      }
      goto LABEL_16;
    }
  }
  v26 = 1;
  if ( !v11 )
    v11 = &v25;
  v25 = 0;
  v14 = sub_33D1410(v12, v11);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  v6 = *(_QWORD **)(a1 + 40);
  v15 = v6;
  if ( v14 )
    goto LABEL_33;
LABEL_16:
  v7 = *(_DWORD *)(a3 + 8);
LABEL_4:
  v9 = v6[5];
  if ( *(_DWORD *)(v9 + 24) != v7 )
    return 0;
  v16 = *(_QWORD *)(a3 + 16);
  v23 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v9 + 40));
  *(_QWORD *)v16 = v23.m128i_i64[0];
  *(_DWORD *)(v16 + 8) = v23.m128i_i32[2];
  v17 = *(unsigned __int64 **)(a3 + 24);
  v18 = *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL);
  if ( v18 && ((v19 = *(_DWORD *)(v18 + 24), v19 == 35) || v19 == 11) )
  {
    if ( v17 )
    {
      v22 = *(_QWORD *)(v18 + 96);
      if ( *((_DWORD *)v17 + 2) <= 0x40u && *(_DWORD *)(v22 + 32) <= 0x40u )
      {
        *v17 = *(_QWORD *)(v22 + 24);
        *((_DWORD *)v17 + 2) = *(_DWORD *)(v22 + 32);
      }
      else
      {
        sub_C43990(*(_QWORD *)(a3 + 24), v22 + 24);
      }
    }
  }
  else
  {
    v26 = 1;
    if ( !v17 )
      v17 = &v25;
    v25 = 0;
    v20 = sub_33D1410(v18, v17);
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    if ( !v20 )
      return 0;
  }
  if ( *(_BYTE *)(a3 + 36) && *(_DWORD *)(a3 + 32) != (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v9 + 28))
    || !(unsigned __int8)sub_32657E0(a3 + 40, **(_QWORD **)(a1 + 40)) )
  {
    return 0;
  }
LABEL_29:
  result = 1;
  if ( *(_BYTE *)(a3 + 60) )
    return (*(_DWORD *)(a3 + 56) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 56);
  return result;
}
