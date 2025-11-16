// Function: sub_32390D0
// Address: 0x32390d0
//
__int64 __fastcall sub_32390D0(__int64 a1)
{
  __int64 *v1; // r14
  __int64 result; // rax
  __int64 *i; // r13
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // r15
  unsigned __int8 v9; // al

  v1 = *(__int64 **)(a1 + 2824);
  result = *(unsigned int *)(a1 + 2832);
  for ( i = &v1[result]; i != v1; result = sub_37374D0(v7, v8) )
  {
    while ( 1 )
    {
      v8 = *v1;
      v9 = *(_BYTE *)(*v1 - 16);
      v5 = (v9 & 2) != 0 ? *(_QWORD *)(v8 - 32) : v8 - 16 - 8LL * ((v9 >> 2) & 0xF);
      v6 = sub_3238860(a1, *(_QWORD *)(v5 + 40));
      result = sub_37374D0(v6, v8);
      v7 = *(_QWORD *)(v6 + 408);
      if ( v7 )
      {
        result = *(_QWORD *)(v6 + 80);
        if ( *(_BYTE *)(result + 41) )
          break;
      }
      if ( i == ++v1 )
        return result;
    }
    ++v1;
  }
  return result;
}
