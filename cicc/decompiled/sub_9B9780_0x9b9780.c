// Function: sub_9B9780
// Address: 0x9b9780
//
__int64 __fastcall sub_9B9780(__int64 a1, int *a2, __int64 a3, int a4)
{
  __int64 v4; // r9
  int *v5; // r15
  int *v6; // r13
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  int v10; // ebx
  __int64 v12; // [rsp+8h] [rbp-38h]

  v4 = a1 + 16;
  v5 = &a2[a3];
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  if ( a2 != v5 )
  {
    v6 = a2;
    v8 = 16;
    v9 = 0;
    while ( 1 )
    {
      v10 = *v6;
      if ( a4 <= *v6 )
        v10 = *v6 - a4;
      if ( v9 + 1 > v8 )
      {
        v12 = v4;
        sub_C8D5F0(a1, v4, v9 + 1, 4);
        v9 = *(unsigned int *)(a1 + 8);
        v4 = v12;
      }
      ++v6;
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v9) = v10;
      v9 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v9;
      if ( v5 == v6 )
        break;
      v8 = *(unsigned int *)(a1 + 12);
    }
  }
  return a1;
}
