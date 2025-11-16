// Function: sub_A74D20
// Address: 0xa74d20
//
__int64 __fastcall sub_A74D20(__int64 a1, int a2)
{
  __int64 *v2; // r12
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 *v5; // r14
  __int64 v7[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  if ( !*(_DWORD *)(a1 + 16) )
    return 0;
  v3 = *(unsigned int *)(a1 + 16);
  do
  {
    while ( 1 )
    {
      v4 = v3 >> 1;
      v5 = &v2[v3 >> 1];
      v7[0] = *v5;
      if ( sub_A71840((__int64)v7) || (int)sub_A71AE0(v7) >= a2 )
        break;
      v2 = v5 + 1;
      v3 = v3 - v4 - 1;
      if ( v3 <= 0 )
        goto LABEL_7;
    }
    v3 >>= 1;
  }
  while ( v4 > 0 );
LABEL_7:
  if ( v2 != (__int64 *)(*(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16)) && sub_A71B30(v2, a2) )
    return *v2;
  else
    return 0;
}
