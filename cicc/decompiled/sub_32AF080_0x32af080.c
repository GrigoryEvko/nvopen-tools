// Function: sub_32AF080
// Address: 0x32af080
//
char __fastcall sub_32AF080(int *a1, __int64 a2, __int64 a3)
{
  int v5; // r14d
  __int64 v6; // rdi
  char result; // al
  unsigned int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 *v12; // r8
  __int64 v13; // rdi
  int v14; // eax
  char v15; // r13
  __int64 v16; // rax
  int v17; // eax
  char v18; // r13
  __int64 v19; // rax
  __m128i v20; // [rsp+0h] [rbp-60h]
  __m128i v21; // [rsp+10h] [rbp-50h]
  __int64 v22; // [rsp+28h] [rbp-38h]
  unsigned __int64 v23; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v24; // [rsp+38h] [rbp-28h]

  v5 = *a1;
  if ( !(unsigned __int8)sub_33CB110(*(unsigned int *)(a3 + 24)) )
  {
    v6 = *(unsigned int *)(a3 + 24);
    if ( v5 != (_DWORD)v6 )
      return 0;
LABEL_14:
    sub_33CB110(v6);
    v11 = *((_QWORD *)a1 + 1);
    v21 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a3 + 40));
    *(_QWORD *)v11 = v21.m128i_i64[0];
    *(_DWORD *)(v11 + 8) = v21.m128i_i32[2];
    v12 = (unsigned __int64 *)*((_QWORD *)a1 + 2);
    v13 = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 40LL);
    if ( !v13 || (v14 = *(_DWORD *)(v13 + 24), v14 != 11) && v14 != 35 )
    {
      v24 = 1;
      if ( !v12 )
        v12 = &v23;
      v23 = 0;
      v15 = sub_33D1410(v13, v12);
      if ( v24 > 0x40 && v23 )
        j_j___libc_free_0_0(v23);
      if ( v15 )
        goto LABEL_32;
      v16 = *((_QWORD *)a1 + 1);
      v20 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a3 + 40) + 40LL));
      *(_QWORD *)v16 = v20.m128i_i64[0];
      *(_DWORD *)(v16 + 8) = v20.m128i_i32[2];
      v12 = (unsigned __int64 *)*((_QWORD *)a1 + 2);
      v13 = **(_QWORD **)(a3 + 40);
      if ( !v13 || (v17 = *(_DWORD *)(v13 + 24), v17 != 11) && v17 != 35 )
      {
        v24 = 1;
        if ( !v12 )
          v12 = &v23;
        v23 = 0;
        v18 = sub_33D1410(v13, v12);
        if ( v24 > 0x40 && v23 )
          j_j___libc_free_0_0(v23);
        if ( !v18 )
          return 0;
        goto LABEL_32;
      }
    }
    if ( v12 )
    {
      v19 = *(_QWORD *)(v13 + 96);
      if ( *((_DWORD *)v12 + 2) <= 0x40u && *(_DWORD *)(v19 + 32) <= 0x40u )
      {
        *v12 = *(_QWORD *)(v19 + 24);
        *((_DWORD *)v12 + 2) = *(_DWORD *)(v19 + 32);
      }
      else
      {
        sub_C43990((__int64)v12, v19 + 24);
      }
    }
LABEL_32:
    result = 1;
    if ( *((_BYTE *)a1 + 28) )
      return (a1[6] & *(_DWORD *)(a3 + 28)) == a1[6];
    return result;
  }
  v22 = sub_33CB280(*(unsigned int *)(a3 + 24), ((unsigned __int8)(*(_DWORD *)(a3 + 28) >> 12) ^ 1) & 1);
  if ( !BYTE4(v22) || v5 != (_DWORD)v22 )
    return 0;
  v8 = *(_DWORD *)(a3 + 24);
  v23 = sub_33CB160(v8);
  if ( !BYTE4(v23)
    || (v9 = *(_QWORD *)(a3 + 40) + 40LL * (unsigned int)v23, *(_QWORD *)v9 == *(_QWORD *)(a2 + 16))
    && *(_DWORD *)(v9 + 8) == *(_DWORD *)(a2 + 24)
    || (result = sub_33D1720(*(_QWORD *)v9, 0)) != 0 )
  {
    v23 = sub_33CB1F0(v8);
    if ( BYTE4(v23) )
    {
      v10 = *(_QWORD *)(a3 + 40) + 40LL * (unsigned int)v23;
      if ( *(_QWORD *)(a2 + 32) != *(_QWORD *)v10 || *(_DWORD *)(a2 + 40) != *(_DWORD *)(v10 + 8) )
        return 0;
    }
    v6 = *(unsigned int *)(a3 + 24);
    goto LABEL_14;
  }
  return result;
}
