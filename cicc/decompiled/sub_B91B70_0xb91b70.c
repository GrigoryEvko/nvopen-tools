// Function: sub_B91B70
// Address: 0xb91b70
//
void __fastcall sub_B91B70(__int64 *a1, int a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r15

  v3 = *a1;
  v4 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( v4 != *a1 )
  {
    v7 = a3 + 16;
    do
    {
      while ( *(_DWORD *)v3 != a2 )
      {
        v3 += 16;
        if ( v4 == v3 )
          return;
      }
      v8 = *(unsigned int *)(a3 + 8);
      v9 = *(_QWORD *)(v3 + 8);
      if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, v7, v8 + 1, 8);
        v8 = *(unsigned int *)(a3 + 8);
      }
      v3 += 16;
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v9;
      ++*(_DWORD *)(a3 + 8);
    }
    while ( v4 != v3 );
  }
}
