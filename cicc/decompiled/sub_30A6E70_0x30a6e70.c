// Function: sub_30A6E70
// Address: 0x30a6e70
//
__int64 __fastcall sub_30A6E70(__int64 (__fastcall ***a1)(__int64), __int64 a2)
{
  __int64 v2; // r15
  __int64 (__fastcall **v4)(__int64); // rax
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 i; // r12
  __int64 (__fastcall **v10)(__int64); // rax
  __int64 v11; // rsi

  v2 = a2 + 176;
  v4 = *a1;
  v5 = (__int64)(*a1)[1];
  result = (*v4)(v5);
  v8 = *(_QWORD *)(a2 + 192);
  if ( v8 != a2 + 176 )
  {
    do
    {
      for ( i = *(_QWORD *)(v8 + 64); v8 + 48 != i; i = sub_220EEE0(i) )
      {
        v10 = a1[1];
        v11 = i + 40;
        if ( !v10[2] )
          sub_4263D6(v5, v11, v7);
        ((void (__fastcall *)(__int64 (__fastcall **)(__int64), __int64))v10[3])(a1[1], v11);
        v5 = i;
      }
      v5 = v8;
      result = sub_220EEE0(v8);
      v8 = result;
    }
    while ( v2 != result );
  }
  return result;
}
