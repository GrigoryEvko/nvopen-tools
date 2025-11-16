// Function: sub_1FE6610
// Address: 0x1fe6610
//
__int64 __fastcall sub_1FE6610(size_t *a1, unsigned __int64 a2, int a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  int v8; // r10d
  unsigned int i; // r9d
  __int64 v10; // rdx
  unsigned int v11; // r9d
  unsigned __int32 v12; // r13d
  unsigned int v15; // edx
  __int64 v16; // r8
  size_t v17; // rdi
  unsigned __int8 *v18; // r9
  __int64 v19; // rsi
  __int64 (__fastcall *v20)(__int64, unsigned __int8); // rax
  __int64 v21; // rsi
  __int64 *v22; // r14
  __int64 v23; // r15
  __int64 v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rax
  size_t v27; // [rsp+8h] [rbp-68h]
  __m128i v28; // [rsp+10h] [rbp-60h] BYREF
  __int64 v29; // [rsp+20h] [rbp-50h]
  __int64 v30; // [rsp+28h] [rbp-48h]
  __int64 v31; // [rsp+30h] [rbp-40h]

  if ( *(_WORD *)(a2 + 24) == 0xFFF6 )
  {
    v12 = sub_1FE65C0((__int64)a1, a2, a3);
    if ( !v12 )
    {
      v17 = a1[4];
      v18 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * v15);
      v19 = *v18;
      v20 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v17 + 288LL);
      if ( v20 == sub_1D45FB0 )
        v21 = *(_QWORD *)(v17 + 8 * v19 + 120);
      else
        v21 = v20(v17, v19);
      v12 = sub_1E6B9A0(a1[1], v21, (unsigned __int8 *)byte_3F871B3, 0, v16, (int)v18);
    }
    v22 = (__int64 *)a1[6];
    v23 = *(_QWORD *)(a1[5] + 56);
    v27 = a1[5];
    v24 = (__int64)sub_1E0B640(v23, *(_QWORD *)(a1[2] + 8) + 576LL, (__int64 *)(a2 + 72), 0);
    sub_1DD5BA0((__int64 *)(v27 + 16), v24);
    v25 = *v22;
    v26 = *(_QWORD *)v24;
    *(_QWORD *)(v24 + 8) = v22;
    v25 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v24 = v25 | v26 & 7;
    *(_QWORD *)(v25 + 8) = v24;
    *v22 = v24 | *v22 & 7;
    v28.m128i_i64[0] = 0x10000000;
    v29 = 0;
    v28.m128i_i32[2] = v12;
    v30 = 0;
    v31 = 0;
    sub_1E1A9C0(v24, v23, &v28);
  }
  else
  {
    v5 = *(unsigned int *)(a4 + 24);
    v6 = *(_QWORD *)(a4 + 8);
    if ( (_DWORD)v5 )
    {
      v8 = 1;
      for ( i = (v5 - 1) & (((a2 >> 9) ^ (a2 >> 4)) + a3); ; i = (v5 - 1) & v11 )
      {
        v10 = v6 + 24LL * i;
        if ( a2 == *(_QWORD *)v10 )
        {
          if ( *(_DWORD *)(v10 + 8) == a3 )
            return *(unsigned __int32 *)(v10 + 16);
        }
        else if ( !*(_QWORD *)v10 && *(_DWORD *)(v10 + 8) == -1 )
        {
          break;
        }
        v11 = v8 + i;
        ++v8;
      }
    }
    v10 = v6 + 24 * v5;
    return *(unsigned __int32 *)(v10 + 16);
  }
  return v12;
}
