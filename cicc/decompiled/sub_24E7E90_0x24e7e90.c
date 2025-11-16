// Function: sub_24E7E90
// Address: 0x24e7e90
//
_QWORD *__fastcall sub_24E7E90(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  int v8; // eax
  __int64 v9; // rax
  _QWORD *v10; // rbx
  __m128i v11; // xmm2
  __int64 v12; // rax
  __m128i v13; // xmm1
  __int64 v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v18; // rax
  __m128i v19; // xmm0
  __int64 v20; // rax
  __m128i v21; // xmm3
  __m128i v22; // [rsp+0h] [rbp-70h] BYREF
  void (__fastcall *v23)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-60h]
  __int64 v24; // [rsp+18h] [rbp-58h]

  v6 = *a3;
  v7 = *(_QWORD *)(v6 - 32);
  if ( !v7 || *(_BYTE *)v7 || *(_QWORD *)(v7 + 24) != *(_QWORD *)(v6 + 80) )
    BUG();
  if ( *(_DWORD *)(v7 + 36) != 40 )
  {
    v8 = *((_DWORD *)a3 + 70);
    if ( v8 == 2 )
    {
      sub_F99F40((__int64)&v22, a4);
      v20 = sub_22077B0(0x38u);
      v10 = (_QWORD *)v20;
      if ( v20 )
      {
        v21 = _mm_loadu_si128(&v22);
        *(_QWORD *)(v20 + 8) = a2;
        *(_QWORD *)(v20 + 16) = a3;
        *(__m128i *)(v20 + 24) = v21;
        goto LABEL_27;
      }
LABEL_20:
      if ( v23 )
        v23(&v22, &v22, 3);
      goto LABEL_22;
    }
    if ( v8 > 2 )
    {
      if ( v8 != 3 )
        goto LABEL_29;
      sub_F99F40((__int64)&v22, a4);
      v12 = sub_22077B0(0x38u);
      v10 = (_QWORD *)v12;
      if ( !v12 )
        goto LABEL_20;
      v13 = _mm_loadu_si128(&v22);
      *(_QWORD *)(v12 + 8) = a2;
      *(_QWORD *)(v12 + 16) = a3;
      *(__m128i *)(v12 + 24) = v13;
      *(_QWORD *)(v12 + 40) = v23;
      *(_QWORD *)(v12 + 48) = v24;
      *(_QWORD *)v12 = &unk_4A16AE8;
    }
    else
    {
      if ( v8 )
      {
        if ( v8 == 1 )
        {
          sub_F99F40((__int64)&v22, a4);
          v9 = sub_22077B0(0x38u);
          v10 = (_QWORD *)v9;
          if ( v9 )
          {
            v11 = _mm_loadu_si128(&v22);
            *(_QWORD *)(v9 + 8) = a2;
            *(_QWORD *)(v9 + 16) = a3;
            *(__m128i *)(v9 + 24) = v11;
LABEL_27:
            v10[5] = v23;
            v10[6] = v24;
            *v10 = &unk_4A16B20;
            goto LABEL_22;
          }
          goto LABEL_20;
        }
LABEL_29:
        BUG();
      }
      sub_F99F40((__int64)&v22, a4);
      v18 = sub_22077B0(0x38u);
      v10 = (_QWORD *)v18;
      if ( !v18 )
        goto LABEL_20;
      v19 = _mm_loadu_si128(&v22);
      *(_QWORD *)(v18 + 8) = a2;
      *(_QWORD *)(v18 + 16) = a3;
      *(__m128i *)(v18 + 24) = v19;
      *(_QWORD *)(v18 + 40) = v23;
      *(_QWORD *)(v18 + 48) = v24;
      *(_QWORD *)v18 = &unk_4A16AB0;
    }
LABEL_22:
    *a1 = v10;
    return a1;
  }
  v14 = *(_QWORD *)(v6 + 32 * (2LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
  v15 = *(_QWORD **)(v14 + 24);
  if ( *(_DWORD *)(v14 + 32) > 0x40u )
    v15 = (_QWORD *)*v15;
  if ( *(_DWORD *)(a5 + 8) <= (unsigned int)v15 )
    BUG();
  v16 = *(_QWORD *)a5 + 32LL * (unsigned int)v15;
  if ( !*(_QWORD *)(v16 + 16) )
    sub_4263D6(a1, a4, v15);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64 *))(v16 + 24))(a1, v16, a2, a3);
  return a1;
}
