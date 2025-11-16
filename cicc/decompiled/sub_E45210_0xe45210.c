// Function: sub_E45210
// Address: 0xe45210
//
void __fastcall sub_E45210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 i; // rbx
  __int64 v7; // rax

  for ( i = *(_QWORD *)(a2 + 56); a2 + 48 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(i - 24) - 30 <= 0xA )
      break;
    v7 = *(unsigned int *)(a1 + 8);
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v7 + 1, 8u, a5, a6);
      v7 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v7) = i - 24;
    ++*(_DWORD *)(a1 + 8);
  }
}
