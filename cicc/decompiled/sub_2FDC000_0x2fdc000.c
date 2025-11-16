// Function: sub_2FDC000
// Address: 0x2fdc000
//
__int64 __fastcall sub_2FDC000(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  __int64 v5; // r9
  int v6; // ecx
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // r8
  int v9; // eax
  __int64 result; // rax
  unsigned int *v11; // rdx
  unsigned int *i; // rdi
  unsigned int v13; // ecx
  __int64 v14; // r13

  v4 = *(_DWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16)) + 16);
  v6 = *(_DWORD *)(a3 + 64) & 0x3F;
  if ( v6 )
    *(_QWORD *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) - 8) &= ~(-1LL << v6);
  v7 = *(unsigned int *)(a3 + 8);
  *(_DWORD *)(a3 + 64) = v4;
  v8 = (unsigned int)(v4 + 63) >> 6;
  if ( v8 != v7 )
  {
    if ( v8 >= v7 )
    {
      v14 = v8 - v7;
      if ( v8 > *(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v8, 8u, v8, v5);
        v7 = *(unsigned int *)(a3 + 8);
      }
      if ( 8 * v14 )
      {
        memset((void *)(*(_QWORD *)a3 + 8 * v7), 0, 8 * v14);
        LODWORD(v7) = *(_DWORD *)(a3 + 8);
      }
      v4 = *(_DWORD *)(a3 + 64);
      *(_DWORD *)(a3 + 8) = v14 + v7;
    }
    else
    {
      *(_DWORD *)(a3 + 8) = (unsigned int)(v4 + 63) >> 6;
    }
  }
  v9 = v4 & 0x3F;
  if ( v9 )
    *(_QWORD *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) - 8) &= ~(-1LL << v9);
  result = *(_QWORD *)(a2 + 48);
  if ( *(_BYTE *)(result + 120) )
  {
    v11 = *(unsigned int **)(result + 96);
    for ( i = *(unsigned int **)(result + 104); i != v11; *(_QWORD *)(*(_QWORD *)a3 + 8 * result) |= 1LL << v13 )
    {
      v13 = *v11;
      v11 += 3;
      result = v13 >> 6;
    }
  }
  return result;
}
