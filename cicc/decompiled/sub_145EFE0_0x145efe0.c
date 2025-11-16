// Function: sub_145EFE0
// Address: 0x145efe0
//
__int64 __fastcall sub_145EFE0(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  __int64 v14; // [rsp+8h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  v4 = ((((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
      | (*((unsigned int *)a1 + 3) + 2LL)
      | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 4;
  v5 = ((v4
       | (((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
       | (*((unsigned int *)a1 + 3) + 2LL)
       | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 8)
     | v4
     | (((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
     | (*((unsigned int *)a1 + 3) + 2LL)
     | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1);
  v6 = (v5 | (v5 >> 16) | HIDWORD(v5)) + 1;
  if ( v6 >= a2 )
    v3 = v6;
  v7 = v3;
  if ( v3 > 0xFFFFFFFF )
    v7 = 0xFFFFFFFFLL;
  v14 = malloc(104 * v7);
  if ( !v14 )
    sub_16BD1C0("Allocation failed");
  v8 = *a1 + 104LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v8 )
  {
    v9 = v14;
    v10 = *a1;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = *(_QWORD *)v10;
        *(_QWORD *)(v9 + 8) = *(_QWORD *)(v10 + 8);
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v10 + 16);
        *(_BYTE *)(v9 + 24) = *(_BYTE *)(v10 + 24);
        sub_16CCEE0(v9 + 32, v9 + 72, 4, v10 + 32);
      }
      v10 += 104LL;
      v9 += 104;
    }
    while ( v8 != v10 );
    v11 = *a1;
    v8 = *a1 + 104LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v8 )
    {
      do
      {
        v8 -= 104LL;
        v12 = *(_QWORD *)(v8 + 48);
        if ( v12 != *(_QWORD *)(v8 + 40) )
          _libc_free(v12);
      }
      while ( v8 != v11 );
      v8 = *a1;
    }
  }
  if ( (unsigned __int64 *)v8 != a1 + 2 )
    _libc_free(v8);
  *((_DWORD *)a1 + 3) = v7;
  *a1 = v14;
  return v14;
}
