// Function: sub_2B10250
// Address: 0x2b10250
//
__int64 __fastcall sub_2B10250(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r11
  __int64 v8; // r10
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r12
  int i; // edx
  int v14; // r13d

  if ( a1 == a2 )
    return a3;
  while ( 1 )
  {
    v6 = *(unsigned int *)(a3 + 24);
    v7 = *a1;
    v8 = *(_QWORD *)(a3 + 8);
    if ( (_DWORD)v6 )
      break;
    a1 += 4;
    if ( a2 == a1 )
      return a3;
  }
LABEL_5:
  v9 = (v6 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( v7 != *v10 )
  {
    for ( i = 1; ; i = v14 )
    {
      if ( v11 == -4096 )
        goto LABEL_8;
      v14 = i + 1;
      v9 = (v6 - 1) & (i + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v7 == *v10 )
        break;
    }
  }
  if ( v10 != (__int64 *)(v8 + 16 * v6) )
    *(_QWORD *)(*(_QWORD *)(a4 + 2232) + 32LL * *((unsigned int *)v10 + 2) + 8) = 0;
LABEL_8:
  while ( 1 )
  {
    a1 += 4;
    if ( a2 == a1 )
      return a3;
    v6 = *(unsigned int *)(a3 + 24);
    v7 = *a1;
    v8 = *(_QWORD *)(a3 + 8);
    if ( (_DWORD)v6 )
      goto LABEL_5;
  }
}
