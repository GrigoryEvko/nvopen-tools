// Function: sub_1F70880
// Address: 0x1f70880
//
__int64 *__fastcall sub_1F70880(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *v6; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // r15
  const void ***v11; // rcx
  __int64 v12; // rsi
  __int64 *v13; // r10
  int v14; // r8d
  __int64 *result; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned int v18; // esi
  __int128 v19; // [rsp-20h] [rbp-80h]
  __int128 v20; // [rsp-10h] [rbp-70h]
  int v21; // [rsp+0h] [rbp-60h]
  const void ***v22; // [rsp+8h] [rbp-58h]
  __int64 *v23; // [rsp+10h] [rbp-50h]
  __int64 *v24; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+20h] [rbp-40h] BYREF
  int v27; // [rsp+28h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 32);
  v7 = *v6;
  v8 = v6[1];
  v9 = v6[5];
  v10 = v6[6];
  if ( !sub_1D185B0(v6[10]) )
    return 0;
  v11 = *(const void ****)(a2 + 40);
  if ( *(_BYTE *)(a1 + 24) )
  {
    v16 = *(unsigned __int8 *)v11;
    v17 = *(_QWORD *)(a1 + 8);
    v18 = 1;
    if ( (_BYTE)v16 != 1 )
    {
      if ( !(_BYTE)v16 )
        return 0;
      v18 = (unsigned __int8)v16;
      if ( !*(_QWORD *)(v17 + 8 * v16 + 120) )
        return 0;
    }
    if ( (*(_BYTE *)(v17 + 259LL * v18 + 2495) & 0xFB) != 0 )
      return 0;
  }
  v12 = *(_QWORD *)(a2 + 72);
  v13 = *(__int64 **)a1;
  v14 = *(_DWORD *)(a2 + 60);
  v26 = v12;
  if ( v12 )
  {
    v21 = v14;
    v22 = v11;
    v23 = v13;
    sub_1623A60((__int64)&v26, v12, 2);
    v14 = v21;
    v11 = v22;
    v13 = v23;
  }
  *((_QWORD *)&v20 + 1) = v10;
  *(_QWORD *)&v20 = v9;
  *((_QWORD *)&v19 + 1) = v8;
  *(_QWORD *)&v19 = v7;
  v27 = *(_DWORD *)(a2 + 64);
  result = sub_1D37440(v13, 73, (__int64)&v26, v11, v14, (__int64)&v26, a3, a4, a5, v19, v20);
  if ( v26 )
  {
    v24 = result;
    sub_161E7C0((__int64)&v26, v26);
    return v24;
  }
  return result;
}
