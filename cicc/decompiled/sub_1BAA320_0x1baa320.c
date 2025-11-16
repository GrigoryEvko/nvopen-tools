// Function: sub_1BAA320
// Address: 0x1baa320
//
__int64 __fastcall sub_1BAA320(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  char v4; // al
  char v6; // bl
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *i; // rdx
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rax
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r14
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 result; // rax
  __int64 v27; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v28[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 (__fastcall *v29)(const __m128i **, const __m128i *, int); // [rsp+20h] [rbp-30h]
  bool (__fastcall *v30)(_QWORD **, int *); // [rsp+28h] [rbp-28h]

  v4 = *(_BYTE *)(a2 + 16);
  v27 = a2;
  if ( (unsigned __int8)(v4 - 54) > 1u )
    return 0;
  v28[0] = a1;
  v28[1] = &v27;
  v30 = sub_1B995D0;
  v29 = sub_1B8E1F0;
  v6 = sub_1B932A0((__int64)v28, a3, (__int64)a3);
  if ( v29 )
    v29((const __m128i **)v28, (const __m128i *)v28, 3);
  if ( !v6 )
    return 0;
  v7 = *(_QWORD *)(a1 + 24);
  v8 = *(_QWORD *)(v7 + 504);
  if ( v8 == *(_QWORD *)(v7 + 496) )
    v9 = *(unsigned int *)(v7 + 516);
  else
    v9 = *(unsigned int *)(v7 + 512);
  v10 = (_QWORD *)(v8 + 8 * v9);
  v11 = sub_15CC2D0(v7 + 488, v27);
  v12 = *(_QWORD *)(v7 + 504);
  if ( v12 == *(_QWORD *)(v7 + 496) )
    v13 = *(unsigned int *)(v7 + 516);
  else
    v13 = *(unsigned int *)(v7 + 512);
  for ( i = (_QWORD *)(v12 + 8 * v13); i != v11; ++v11 )
  {
    if ( *v11 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  if ( v11 != v10 )
  {
    v15 = sub_1BA9430(a1, *(_QWORD *)(v27 + 40), a4);
    v16 = sub_22077B0(56);
    v17 = v16;
    if ( v16 )
    {
      *(_QWORD *)(v16 + 8) = 0;
      *(_QWORD *)(v16 + 16) = 0;
      *(_BYTE *)(v16 + 24) = 7;
      *(_QWORD *)(v16 + 32) = 0;
      *(_QWORD *)(v16 + 48) = 0;
      *(_QWORD *)v16 = &unk_49F7038;
      *(_QWORD *)(v16 + 40) = v27;
      if ( v15 )
      {
        v18 = sub_22077B0(72);
        v21 = v18;
        if ( v18 )
        {
          *(_BYTE *)v18 = 1;
          *(_QWORD *)(v18 + 8) = v18 + 24;
          *(_QWORD *)(v18 + 16) = 0x100000000LL;
          *(_QWORD *)(v18 + 40) = v18 + 56;
          *(_QWORD *)(v18 + 32) = 0;
          *(_QWORD *)(v18 + 56) = v15;
          *(_QWORD *)(v18 + 48) = 0x200000001LL;
          v22 = *(unsigned int *)(v15 + 16);
          if ( (unsigned int)v22 >= *(_DWORD *)(v15 + 20) )
          {
            sub_16CD150(v15 + 8, (const void *)(v15 + 24), 0, 8, v19, v20);
            v22 = *(unsigned int *)(v15 + 16);
          }
          *(_QWORD *)(*(_QWORD *)(v15 + 8) + 8 * v22) = v21;
          ++*(_DWORD *)(v15 + 16);
        }
        v23 = *(_QWORD *)(v17 + 48);
        *(_QWORD *)(v17 + 48) = v21;
        if ( v23 )
        {
          v24 = *(_QWORD *)(v23 + 40);
          if ( v24 != v23 + 56 )
            _libc_free(v24);
          v25 = *(_QWORD *)(v23 + 8);
          if ( v25 != v23 + 24 )
            _libc_free(v25);
          j_j___libc_free_0(v23, 72);
        }
      }
      return v17;
    }
    return 0;
  }
  result = sub_22077B0(56);
  v17 = result;
  if ( !result )
    return v17;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = 0;
  *(_BYTE *)(result + 24) = 7;
  *(_QWORD *)(result + 32) = 0;
  *(_QWORD *)(result + 48) = 0;
  *(_QWORD *)result = &unk_49F7038;
  *(_QWORD *)(result + 40) = v27;
  return result;
}
