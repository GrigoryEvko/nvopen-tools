// Function: sub_B92230
// Address: 0xb92230
//
__int64 __fastcall sub_B92230(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 *v4; // rdi
  __int64 result; // rax
  __int64 *v6; // r15
  __int64 *v7; // rbx
  __int64 v8; // r14
  __int64 *v9; // [rsp+10h] [rbp-50h] BYREF
  __int64 v10; // [rsp+18h] [rbp-48h]
  _BYTE v11[64]; // [rsp+20h] [rbp-40h] BYREF

  v3 = 0;
  v10 = 0x100000000LL;
  v9 = (__int64 *)v11;
  sub_B91D10(a1, 0, (__int64)&v9);
  v4 = v9;
  result = (unsigned int)v10;
  v6 = &v9[(unsigned int)v10];
  if ( v6 != v9 )
  {
    result = *(unsigned int *)(a2 + 8);
    v7 = v9;
    v3 = a2 + 16;
    do
    {
      v8 = *v7;
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, v3, result + 1, 8);
        result = *(unsigned int *)(a2 + 8);
      }
      ++v7;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v8;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
    }
    while ( v6 != v7 );
    v4 = v9;
  }
  if ( v4 != (__int64 *)v11 )
    return _libc_free(v4, v3);
  return result;
}
