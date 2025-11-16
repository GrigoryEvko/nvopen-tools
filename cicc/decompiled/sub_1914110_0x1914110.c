// Function: sub_1914110
// Address: 0x1914110
//
__int64 *__fastcall sub_1914110(__int64 a1)
{
  __int64 *result; // rax
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  int v5; // eax
  __int64 *v6; // [rsp+8h] [rbp-48h]
  __int64 *v7; // [rsp+10h] [rbp-40h]
  __int64 i; // [rsp+18h] [rbp-38h]

  result = *(__int64 **)(a1 + 88);
  v6 = result;
  v7 = *(__int64 **)(a1 + 80);
  if ( v7 != result )
  {
    do
    {
      v2 = *v7;
      v3 = *(_QWORD *)(*v7 + 48);
      for ( i = *v7 + 40; i != v3; v3 = *(_QWORD *)(v3 + 8) )
      {
        v4 = 0;
        if ( v3 )
          v4 = v3 - 24;
        v5 = sub_1911FD0(a1 + 152, v4);
        sub_1910810(a1, v5, v4, v2);
      }
      result = ++v7;
    }
    while ( v6 != v7 );
  }
  return result;
}
