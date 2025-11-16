// Function: sub_1A95180
// Address: 0x1a95180
//
__int64 __fastcall sub_1A95180(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r15
  _QWORD *v6; // rax
  __int64 v7; // r12
  __int64 v8; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)a1;
  result = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  v8 = result;
  if ( *(_QWORD *)a1 != result )
  {
    do
    {
      v4 = *v2;
      v5 = sub_1599A20(*(__int64 ***)(*v2 + 56));
      v6 = sub_1648A60(64, 2u);
      v7 = (__int64)v6;
      if ( v6 )
        sub_15F9650((__int64)v6, v5, v4, 0, 0);
      ++v2;
      result = sub_15F2120(v7, a2);
    }
    while ( (__int64 *)v8 != v2 );
  }
  return result;
}
