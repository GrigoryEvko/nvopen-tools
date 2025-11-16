// Function: sub_2F70C30
// Address: 0x2f70c30
//
__int64 __fastcall sub_2F70C30(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // r9
  __int64 *v3; // rbx
  __int64 result; // rax
  __int64 *i; // r14
  __int64 v6; // r13

  sub_2F6DDB0(a1, *(unsigned __int64 **)(a1 + 584), *(unsigned int *)(a1 + 592));
  v3 = *(__int64 **)(a1 + 584);
  result = *(unsigned int *)(a1 + 592);
  for ( i = &v3[result]; i != v3; ++v3 )
  {
    v6 = *v3;
    if ( *v3 )
    {
      result = *(unsigned int *)(a1 + 512);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 516) )
      {
        sub_C8D5F0(a1 + 504, (const void *)(a1 + 520), result + 1, 8u, v1, v2);
        result = *(unsigned int *)(a1 + 512);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 504) + 8 * result) = v6;
      ++*(_DWORD *)(a1 + 512);
    }
  }
  *(_DWORD *)(a1 + 592) = 0;
  return result;
}
