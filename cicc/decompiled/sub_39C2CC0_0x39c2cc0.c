// Function: sub_39C2CC0
// Address: 0x39c2cc0
//
void __fastcall sub_39C2CC0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rcx
  unsigned __int64 *v4; // rbx
  unsigned __int64 *v5; // r13

  v1 = *(unsigned int *)(a1 + 152);
  v2 = *(_QWORD *)(a1 + 144) + 32 * v1 - 32;
  if ( *(_QWORD *)(v2 + 16) == *(_DWORD *)(a1 + 1192) )
  {
    v3 = *(_QWORD *)(a1 + 1456);
    v4 = (unsigned __int64 *)(v3 + 32LL * *(unsigned int *)(a1 + 1464));
    v5 = (unsigned __int64 *)(v3 + 32LL * *(_QWORD *)(v2 + 24));
    if ( v5 != v4 )
    {
      do
      {
        v4 -= 4;
        if ( (unsigned __int64 *)*v4 != v4 + 2 )
          j_j___libc_free_0(*v4);
      }
      while ( v5 != v4 );
      LODWORD(v1) = *(_DWORD *)(a1 + 152);
      v3 = *(_QWORD *)(a1 + 1456);
    }
    *(_DWORD *)(a1 + 152) = v1 - 1;
    *(_DWORD *)(a1 + 1464) = ((__int64)v5 - v3) >> 5;
  }
}
