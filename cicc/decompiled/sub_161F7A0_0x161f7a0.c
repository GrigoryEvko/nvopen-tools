// Function: sub_161F7A0
// Address: 0x161f7a0
//
void __fastcall sub_161F7A0(__int64 *a1, int a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v7; // rsi
  __int64 v8; // r15
  __int64 v9; // rax

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
      v8 = *(_QWORD *)(v3 + 8);
      v9 = *(unsigned int *)(a3 + 8);
      if ( (unsigned int)v9 >= *(_DWORD *)(a3 + 12) )
      {
        sub_16CD150(a3, v7, 0, 8);
        v9 = *(unsigned int *)(a3 + 8);
      }
      v3 += 16;
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v8;
      ++*(_DWORD *)(a3 + 8);
    }
    while ( v4 != v3 );
  }
}
