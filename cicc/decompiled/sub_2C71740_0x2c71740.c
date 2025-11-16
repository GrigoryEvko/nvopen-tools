// Function: sub_2C71740
// Address: 0x2c71740
//
__int64 __fastcall sub_2C71740(unsigned int *a1, __int64 a2, __int64 *a3)
{
  unsigned __int64 v5; // rsi
  _BYTE *v6; // rax
  __int64 result; // rax
  _BYTE *v8; // rdx
  __int64 *v9[6]; // [rsp+0h] [rbp-30h] BYREF

  v5 = *a1;
  if ( a3 )
    sub_2C6EAD0(a2, *(__int64 **)(a3[5] + 8 * v5));
  else
    sub_CB59D0(a2, v5);
  v6 = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == v6 )
  {
    a2 = sub_CB6200(a2, (unsigned __int8 *)":", 1u);
  }
  else
  {
    *v6 = 58;
    ++*(_QWORD *)(a2 + 32);
  }
  v9[1] = a3;
  v9[0] = (__int64 *)(a1 + 2);
  result = sub_2C71280(a2, v9);
  v8 = *(_BYTE **)(result + 32);
  if ( *(_BYTE **)(result + 24) == v8 )
    return sub_CB6200(result, (unsigned __int8 *)"\n", 1u);
  *v8 = 10;
  ++*(_QWORD *)(result + 32);
  return result;
}
