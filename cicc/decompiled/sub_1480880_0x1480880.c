// Function: sub_1480880
// Address: 0x1480880
//
__int64 __fastcall sub_1480880(_QWORD *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 *v4; // r14
  __int64 *i; // rbx
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 *v11; // [rsp+10h] [rbp-50h] BYREF
  __int64 v12; // [rsp+18h] [rbp-48h]
  _BYTE v13[64]; // [rsp+20h] [rbp-40h] BYREF

  v4 = *(__int64 **)a2;
  v11 = (__int64 *)v13;
  v12 = 0x200000000LL;
  for ( i = &v4[*(unsigned int *)(a2 + 8)]; i != v4; LODWORD(v12) = v12 + 1 )
  {
    v6 = sub_1480810((__int64)a1, *v4);
    v7 = (unsigned int)v12;
    if ( (unsigned int)v12 >= HIDWORD(v12) )
    {
      sub_16CD150(&v11, v13, 0, 8);
      v7 = (unsigned int)v12;
    }
    ++v4;
    v11[v7] = v6;
  }
  v8 = sub_147A3C0(a1, &v11, a3, a4);
  v9 = sub_1480810((__int64)a1, v8);
  if ( v11 != (__int64 *)v13 )
    _libc_free((unsigned __int64)v11);
  return v9;
}
