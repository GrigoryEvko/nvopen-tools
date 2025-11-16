// Function: sub_35E4D80
// Address: 0x35e4d80
//
__int64 __fastcall sub_35E4D80(__int64 a1, __int64 a2, unsigned int a3, int a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // edi
  unsigned int v8; // r13d
  int v9; // r14d
  unsigned int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // rcx

  v7 = a4 * *(_DWORD *)(a2 + 6436);
  v8 = v7 / 0x64;
  if ( a3 && v8 >= a3 )
    v9 = v8 / a3;
  else
    v9 = 1;
  v10 = 0;
  v11 = 0;
  *(_QWORD *)a1 = a1 + 16;
  v12 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0xC00000000LL;
  if ( v7 > 0x63 )
  {
    while ( 1 )
    {
      *(_DWORD *)(v12 + 4 * v11) = v10;
      v10 += v9;
      v11 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v11;
      if ( v8 <= v10 )
        break;
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v11 + 1, 4u, v11 + 1, a6);
        v11 = *(unsigned int *)(a1 + 8);
      }
      v12 = *(_QWORD *)a1;
    }
  }
  return a1;
}
