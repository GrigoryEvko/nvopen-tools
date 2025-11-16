// Function: sub_9E19F0
// Address: 0x9e19f0
//
__int64 __fastcall sub_9E19F0(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  int *v5; // rbx
  int *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  if ( a4 )
  {
    v5 = a3;
    v6 = &a3[2 * a4];
    sub_C8D5F0(a1, a1 + 16, a4, 8);
    for ( ; v6 != v5; ++*(_DWORD *)(a1 + 8) )
    {
      v7 = sub_9E1590(a2, *v5);
      v8 = *(unsigned int *)(a1 + 8);
      if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v9 = v7;
        sub_C8D5F0(a1, a1 + 16, v8 + 1, 8);
        v8 = *(unsigned int *)(a1 + 8);
        v7 = v9;
      }
      v5 += 2;
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v8) = v7;
    }
  }
  return a1;
}
