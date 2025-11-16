// Function: sub_27AFD10
// Address: 0x27afd10
//
__int64 __fastcall sub_27AFD10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 *v7; // r13
  __int64 v8; // rax
  __int64 v9; // r14

  v6 = *(unsigned int *)(a1 + 40);
  v7 = *(__int64 **)(a1 + 32);
  if ( *(_DWORD *)(a1 + 40) )
  {
    v8 = *(unsigned int *)(a2 + 8);
    do
    {
      v9 = *v7;
      if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v8 + 1, 8u, a5, a6);
        v8 = *(unsigned int *)(a2 + 8);
      }
      ++v7;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v8) = v9;
      v8 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v8;
      --v6;
    }
    while ( v6 );
  }
  return a2;
}
