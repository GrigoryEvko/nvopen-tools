// Function: sub_2794A80
// Address: 0x2794a80
//
__int64 *__fastcall sub_2794A80(__int64 a1)
{
  __int64 *result; // rax
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  int v5; // eax
  __int64 *v6; // [rsp+8h] [rbp-48h]
  __int64 *v7; // [rsp+10h] [rbp-40h]
  __int64 i; // [rsp+18h] [rbp-38h]

  result = *(__int64 **)(a1 + 80);
  v7 = result;
  v6 = &result[*(unsigned int *)(a1 + 88)];
  if ( v6 != result )
  {
    do
    {
      v2 = *v7;
      v3 = *(_QWORD *)(*v7 + 56);
      for ( i = *v7 + 48; i != v3; v3 = *(_QWORD *)(v3 + 8) )
      {
        v4 = 0;
        if ( v3 )
          v4 = v3 - 24;
        v5 = sub_2792F80(a1 + 136, v4);
        sub_27915B0(a1 + 352, v5, v4, v2);
      }
      result = ++v7;
    }
    while ( v6 != v7 );
  }
  return result;
}
