// Function: sub_107D930
// Address: 0x107d930
//
__int64 __fastcall sub_107D930(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v7; // r15
  __int64 v8; // rsi
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rbx
  char **v14; // r12
  char **v15; // r14
  __int64 v16; // rdx
  char **v17; // rbx
  char **v18; // rdi
  int v19; // ebx
  __int64 v21; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = (char **)(a1 + 16);
  v8 = a1 + 16;
  v21 = sub_C8D7D0(a1, a1 + 16, a2, 0x40u, v22, a6);
  v13 = v21;
  v14 = (char **)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
  if ( *(char ***)a1 != v14 )
  {
    v15 = *(char ***)a1;
    do
    {
      if ( v13 )
      {
        *(_DWORD *)(v13 + 8) = 0;
        *(_QWORD *)v13 = v13 + 16;
        *(_DWORD *)(v13 + 12) = 1;
        v16 = *((unsigned int *)v15 + 2);
        if ( (_DWORD)v16 )
        {
          v8 = (__int64)v15;
          sub_1077380(v13, v15, v16, v10, v11, v12);
        }
        *(_DWORD *)(v13 + 32) = 0;
        *(_QWORD *)(v13 + 24) = v13 + 40;
        *(_DWORD *)(v13 + 36) = 4;
        if ( *((_DWORD *)v15 + 8) )
        {
          v8 = (__int64)(v15 + 3);
          sub_1077380(v13 + 24, v15 + 3, v13 + 40, v10, v11, v12);
        }
        *(_DWORD *)(v13 + 56) = *((_DWORD *)v15 + 14);
        *(_DWORD *)(v13 + 60) = *((_DWORD *)v15 + 15);
      }
      v15 += 8;
      v13 += 64;
    }
    while ( v14 != v15 );
    v17 = *(char ***)a1;
    v14 = (char **)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
    if ( *(char ***)a1 != v14 )
    {
      do
      {
        v14 -= 8;
        v18 = (char **)v14[3];
        if ( v18 != v14 + 5 )
          _libc_free(v18, v8);
        if ( *v14 != (char *)(v14 + 2) )
          _libc_free(*v14, v8);
      }
      while ( v14 != v17 );
      v14 = *(char ***)a1;
    }
  }
  v19 = v22[0];
  if ( v7 != v14 )
    _libc_free(v14, v8);
  *(_DWORD *)(a1 + 12) = v19;
  *(_QWORD *)a1 = v21;
  return v21;
}
