// Function: sub_18E4240
// Address: 0x18e4240
//
__int64 __fastcall sub_18E4240(__int64 a1, __int64 a2, _QWORD *a3, __m128i a4, __m128i a5)
{
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 v15; // rax
  __int64 *v16[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v17[8]; // [rsp+10h] [rbp-40h] BYREF

  v17[0] = sub_1483CF0(a3, a1, a2, a4, a5);
  v16[0] = v17;
  v17[1] = a2;
  v16[1] = (__int64 *)0x200000002LL;
  v6 = sub_147EE30(a3, v16, 0, 0, a4, a5);
  if ( v16[0] != v17 )
    _libc_free((unsigned __int64)v16[0]);
  v7 = sub_14806B0((__int64)a3, v6, a1, 0, 0);
  result = 0;
  if ( !*(_WORD *)(v7 + 24) )
  {
    v9 = *(_QWORD *)(v7 + 32);
    v10 = *(_DWORD *)(v9 + 32);
    v11 = *(__int64 **)(v9 + 24);
    if ( v10 <= 0x40 )
    {
      v12 = (__int64)((_QWORD)v11 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
      if ( !v12 )
      {
LABEL_6:
        v13 = *(_QWORD *)(a2 + 32);
        v14 = *(_DWORD *)(v13 + 32);
        v15 = *(_QWORD *)(v13 + 24);
        if ( v14 > 0x40 )
          return *(_QWORD *)v15;
        else
          return v15 << (64 - (unsigned __int8)v14) >> (64 - (unsigned __int8)v14);
      }
    }
    else
    {
      v12 = *v11;
      if ( !v12 )
        goto LABEL_6;
    }
    result = abs64(v12);
    if ( (result & (result - 1)) != 0 )
      return 0;
  }
  return result;
}
