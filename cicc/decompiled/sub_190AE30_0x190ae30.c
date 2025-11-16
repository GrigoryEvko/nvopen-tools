// Function: sub_190AE30
// Address: 0x190ae30
//
__int64 ***__fastcall sub_190AE30(__int64 a1, __int64 ***a2, __int64 a3, __int64 *a4)
{
  __int64 **v6; // r14
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 ***result; // rax
  __int64 v12; // rsi
  __int64 v13; // r15
  __int64 v14; // [rsp+8h] [rbp-38h]

  v6 = *a2;
  v7 = sub_15F2050((__int64)a2);
  v8 = sub_1632FA0(v7);
  v9 = *(_QWORD *)a1;
  v10 = (*(__int64 *)a1 >> 1) & 3;
  if ( ((*(__int64 *)a1 >> 1) & 3) != 0 )
  {
    if ( v10 == 1 )
    {
      result = (__int64 ***)(v9 & 0xFFFFFFFFFFFFFFF8LL);
      v12 = *(unsigned int *)(a1 + 8);
      v13 = (__int64)result;
      if ( v6 != *result || (_DWORD)v12 )
      {
        v14 = sub_1B730D0(result, v12, v6, a3);
        sub_14191F0(*a4, v13);
        return (__int64 ***)v14;
      }
    }
    else if ( (_DWORD)v10 == 2 )
    {
      return (__int64 ***)sub_1B74740(v9 & 0xFFFFFFFFFFFFFFF8LL, *(unsigned int *)(a1 + 8), v6, a3, v8);
    }
    else
    {
      return (__int64 ***)sub_1599EF0(v6);
    }
  }
  else
  {
    result = (__int64 ***)(v9 & 0xFFFFFFFFFFFFFFF8LL);
    if ( *result != v6 )
      return (__int64 ***)sub_1B725B0(result, *(unsigned int *)(a1 + 8), v6, a3, v8);
  }
  return result;
}
