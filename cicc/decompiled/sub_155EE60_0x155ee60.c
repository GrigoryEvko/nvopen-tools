// Function: sub_155EE60
// Address: 0x155ee60
//
void __fastcall sub_155EE60(__int64 a1, const void *a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 *v4; // rbx
  size_t v5; // rdx
  __int64 i; // r14
  __int64 v7[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a3;
  v4 = (__int64 *)(a1 + 24);
  *(_DWORD *)(a1 + 16) = a3;
  v5 = 8 * a3;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  if ( v5 )
    memmove((void *)(a1 + 24), a2, v5);
  for ( i = a1 + 8LL * v3 + 24; (__int64 *)i != v4; *(_QWORD *)(a1 + 8) |= 1LL << sub_155D410(v7) )
  {
    while ( 1 )
    {
      v7[0] = *v4;
      if ( !sub_155D3E0((__int64)v7) )
        break;
      if ( (__int64 *)i == ++v4 )
        return;
    }
    ++v4;
  }
}
