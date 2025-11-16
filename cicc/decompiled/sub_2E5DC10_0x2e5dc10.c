// Function: sub_2E5DC10
// Address: 0x2e5dc10
//
__int64 __fastcall sub_2E5DC10(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // r13
  __int64 result; // rax
  __int64 *v4; // r15
  __int64 v5; // rdx
  __int64 *i; // r13
  __int64 v8; // rsi
  __int64 v9; // rdx
  _BYTE *v10; // rax
  __int64 v11; // [rsp+8h] [rbp-58h]
  _BYTE v12[16]; // [rsp+10h] [rbp-50h] BYREF
  __int64 (__fastcall *v13)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-40h]
  void (__fastcall *v14)(_BYTE *, __int64); // [rsp+28h] [rbp-38h]

  v2 = *(__int64 **)(*a1 + 8LL);
  result = *(unsigned int *)(*a1 + 16LL);
  v4 = &v2[result];
  if ( v2 != v4 )
  {
    v5 = *v2;
    for ( i = v2 + 1; ; ++i )
    {
      v8 = a1[1];
      sub_2EE72D0(v12, v8, v5);
      if ( !v13 )
        sub_4263D6(v12, v8, v9);
      v14(v12, a2);
      result = (__int64)v13;
      if ( v13 )
        result = v13(v12, v12, 3);
      if ( v4 == i )
        break;
      v5 = *i;
      v10 = *(_BYTE **)(a2 + 32);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(a2 + 24) )
      {
        v11 = *i;
        sub_CB5D20(a2, 32);
        v5 = v11;
      }
      else
      {
        *(_QWORD *)(a2 + 32) = v10 + 1;
        *v10 = 32;
      }
    }
  }
  return result;
}
