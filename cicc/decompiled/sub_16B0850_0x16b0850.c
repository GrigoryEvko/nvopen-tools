// Function: sub_16B0850
// Address: 0x16b0850
//
__int64 __fastcall sub_16B0850(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // r12d
  char v7; // al
  __int64 v8; // r13
  __int64 v9; // rdi
  __int64 v11; // rdi
  int v12; // r8d
  size_t v13; // r15
  unsigned int v14; // r15d
  __int64 v15; // rax

  v5 = 0;
  do
  {
    v8 = a3++;
    ++v5;
    if ( a3 == a2 )
      goto LABEL_4;
    v7 = *(_BYTE *)(a1 + a3);
  }
  while ( v7 == 92 );
  if ( v7 != 34 )
  {
LABEL_4:
    v9 = *(unsigned int *)(a4 + 8);
    if ( v5 > (unsigned __int64)*(unsigned int *)(a4 + 12) - v9 )
    {
      sub_16CD150(a4, a4 + 16, v5 + v9, 1);
      v9 = *(unsigned int *)(a4 + 8);
    }
    memset((void *)(*(_QWORD *)a4 + v9), 92, v5);
    *(_DWORD *)(a4 + 8) += v5;
    return v8;
  }
  v11 = *(unsigned int *)(a4 + 8);
  v12 = v5 >> 1;
  v13 = v5 >> 1;
  if ( v13 > (unsigned __int64)*(unsigned int *)(a4 + 12) - v11 )
  {
    sub_16CD150(a4, a4 + 16, v12 + v11, 1);
    v11 = *(unsigned int *)(a4 + 8);
    v12 = v5 >> 1;
  }
  if ( v12 )
  {
    memset((void *)(*(_QWORD *)a4 + v11), 92, v13);
    LODWORD(v11) = *(_DWORD *)(a4 + 8);
  }
  v14 = v11 + v13;
  *(_DWORD *)(a4 + 8) = v14;
  v15 = v14;
  if ( (v5 & 1) != 0 )
  {
    if ( v14 >= *(_DWORD *)(a4 + 12) )
    {
      sub_16CD150(a4, a4 + 16, 0, 1);
      v15 = *(unsigned int *)(a4 + 8);
    }
    v8 = a3;
    *(_BYTE *)(*(_QWORD *)a4 + v15) = 34;
    ++*(_DWORD *)(a4 + 8);
  }
  return v8;
}
