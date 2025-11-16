// Function: sub_1481AD0
// Address: 0x1481ad0
//
__int64 __fastcall sub_1481AD0(_QWORD *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // rax
  __int64 *v5; // r14
  __int64 *v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // [rsp+8h] [rbp-58h]
  __int64 *v12; // [rsp+10h] [rbp-50h] BYREF
  __int64 i; // [rsp+18h] [rbp-48h]
  _BYTE v14[64]; // [rsp+20h] [rbp-40h] BYREF

  v4 = *(unsigned int *)(a2 + 8);
  v5 = *(__int64 **)a2;
  if ( v4 == 1 )
    return *v5;
  v6 = &v5[v4];
  v12 = (__int64 *)v14;
  for ( i = 0x200000000LL; v6 != v5; LODWORD(i) = i + 1 )
  {
    v7 = sub_1480810((__int64)a1, *v5);
    v8 = (unsigned int)i;
    if ( (unsigned int)i >= HIDWORD(i) )
    {
      sub_16CD150(&v12, v14, 0, 8);
      v8 = (unsigned int)i;
    }
    ++v5;
    v12[v8] = v7;
  }
  v9 = sub_14813B0(a1, &v12, a3, a4);
  result = sub_1480810((__int64)a1, v9);
  if ( v12 != (__int64 *)v14 )
  {
    v11 = result;
    _libc_free((unsigned __int64)v12);
    return v11;
  }
  return result;
}
