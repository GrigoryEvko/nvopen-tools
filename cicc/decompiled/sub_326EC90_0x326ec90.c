// Function: sub_326EC90
// Address: 0x326ec90
//
bool __fastcall sub_326EC90(__int64 a1, __int64 a2)
{
  bool result; // al
  _QWORD *v4; // rax
  int v5; // edx
  __int64 v6; // r13
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // rdi
  int v12; // eax
  char v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // edx
  unsigned int v17; // r8d
  const void **v18; // r13
  const void *v19; // rax
  __m128i v20; // [rsp-68h] [rbp-68h]
  __m128i v21; // [rsp-58h] [rbp-58h]
  const void *v22; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v23; // [rsp-40h] [rbp-40h]
  const void *v24; // [rsp-38h] [rbp-38h] BYREF
  unsigned int v25; // [rsp-30h] [rbp-30h]

  if ( *(_DWORD *)a1 != *(_DWORD *)(a2 + 24) )
    return 0;
  v4 = *(_QWORD **)(a2 + 40);
  v5 = *(_DWORD *)(a1 + 8);
  v6 = *v4;
  if ( v5 == *(_DWORD *)(*v4 + 24LL) )
  {
    v8 = a1 + 24;
    v9 = *(_QWORD *)(v8 - 8);
    v21 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v6 + 40));
    *(_QWORD *)v9 = v21.m128i_i64[0];
    *(_DWORD *)(v9 + 8) = v21.m128i_i32[2];
    v10 = (unsigned __int8)sub_3265630(v8, *(_QWORD *)(*(_QWORD *)(v6 + 40) + 40LL)) == 0;
    v4 = *(_QWORD **)(a2 + 40);
    if ( v10 || *(_BYTE *)(a1 + 44) && *(_DWORD *)(a1 + 40) != (*(_DWORD *)(a1 + 40) & *(_DWORD *)(v6 + 28)) )
      goto LABEL_26;
    v11 = v4[5];
    v23 = 1;
    v22 = 0;
    if ( v11 && ((v12 = *(_DWORD *)(v11 + 24), v12 == 35) || v12 == 11) )
    {
      v15 = *(_QWORD *)(v11 + 96);
      v16 = *(_DWORD *)(v15 + 32);
      if ( v16 <= 0x40 )
      {
        v19 = *(const void **)(v15 + 24);
        v17 = *(_DWORD *)(a1 + 56);
        v23 = v16;
        v18 = (const void **)(a1 + 48);
        v22 = v19;
        if ( v16 != v17 )
        {
LABEL_30:
          if ( v16 >= v17 )
          {
            sub_C449B0((__int64)&v24, v18, v16);
            if ( v25 <= 0x40 )
            {
              v13 = v24 == v22;
              goto LABEL_36;
            }
            v13 = sub_C43C50((__int64)&v24, &v22);
          }
          else
          {
            sub_C449B0((__int64)&v24, &v22, v17);
            if ( *(_DWORD *)(a1 + 56) <= 0x40u )
              v13 = *(_QWORD *)(a1 + 48) == (_QWORD)v24;
            else
              v13 = sub_C43C50((__int64)v18, &v24);
            if ( v25 <= 0x40 )
              goto LABEL_36;
          }
          if ( v24 )
            j_j___libc_free_0_0((unsigned __int64)v24);
LABEL_36:
          if ( v23 <= 0x40 )
            goto LABEL_17;
          goto LABEL_15;
        }
LABEL_43:
        v13 = *(_QWORD *)(a1 + 48) == (_QWORD)v22;
        goto LABEL_17;
      }
      sub_C43990((__int64)&v22, v15 + 24);
      v16 = v23;
    }
    else
    {
      v13 = sub_33D1410(v11, &v22);
      if ( !v13 )
      {
        if ( v23 <= 0x40 )
        {
LABEL_25:
          v4 = *(_QWORD **)(a2 + 40);
LABEL_26:
          v5 = *(_DWORD *)(a1 + 8);
          goto LABEL_4;
        }
        goto LABEL_15;
      }
      v16 = v23;
    }
    v17 = *(_DWORD *)(a1 + 56);
    v18 = (const void **)(a1 + 48);
    if ( v17 != v16 )
      goto LABEL_30;
    if ( v16 > 0x40 )
    {
      v13 = sub_C43C50(a1 + 48, &v22);
LABEL_15:
      if ( v22 )
        j_j___libc_free_0_0((unsigned __int64)v22);
LABEL_17:
      if ( v13 )
        goto LABEL_18;
      goto LABEL_25;
    }
    goto LABEL_43;
  }
LABEL_4:
  v7 = v4[5];
  if ( *(_DWORD *)(v7 + 24) != v5 )
    return 0;
  v14 = *(_QWORD *)(a1 + 16);
  v20 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v7 + 40));
  *(_QWORD *)v14 = v20.m128i_i64[0];
  *(_DWORD *)(v14 + 8) = v20.m128i_i32[2];
  if ( !(unsigned __int8)sub_3265630(a1 + 24, *(_QWORD *)(*(_QWORD *)(v7 + 40) + 40LL))
    || *(_BYTE *)(a1 + 44) && *(_DWORD *)(a1 + 40) != (*(_DWORD *)(a1 + 40) & *(_DWORD *)(v7 + 28)) )
  {
    return 0;
  }
  if ( !(unsigned __int8)sub_3265630(a1 + 48, **(_QWORD **)(a2 + 40)) )
    return 0;
LABEL_18:
  result = 1;
  if ( *(_BYTE *)(a1 + 68) )
    return (*(_DWORD *)(a1 + 64) & *(_DWORD *)(a2 + 28)) == *(_DWORD *)(a1 + 64);
  return result;
}
