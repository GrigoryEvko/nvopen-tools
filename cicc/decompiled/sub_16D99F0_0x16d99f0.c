// Function: sub_16D99F0
// Address: 0x16d99f0
//
__int64 __fastcall sub_16D99F0(unsigned __int64 *a1)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // r12
  _QWORD *v6; // rbx
  unsigned __int64 v7; // r15
  _QWORD *v8; // rbx
  _QWORD *v9; // rdi
  _QWORD *v10; // rdi
  __int64 v12; // [rsp+8h] [rbp-38h]

  v2 = ((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2;
  v3 = ((((v2 | (*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 4)
       | v2
       | (*((unsigned int *)a1 + 3) + 2LL)
       | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 8)
     | ((v2 | (*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 4)
     | v2
     | (*((unsigned int *)a1 + 3) + 2LL)
     | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1);
  v4 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v12 = malloc(80 * v4);
  if ( !v12 )
    sub_16BD1C0("Allocation failed", 1u);
  v5 = *a1 + 80LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v5 )
  {
    v6 = (_QWORD *)v12;
    v7 = *a1;
    do
    {
      if ( v6 )
      {
        *v6 = *(_QWORD *)v7;
        v6[1] = *(_QWORD *)(v7 + 8);
        v6[2] = v6 + 4;
        sub_16D9890(v6 + 2, *(_BYTE **)(v7 + 16), *(_QWORD *)(v7 + 16) + *(_QWORD *)(v7 + 24));
        v6[6] = v6 + 8;
        sub_16D9890(v6 + 6, *(_BYTE **)(v7 + 48), *(_QWORD *)(v7 + 48) + *(_QWORD *)(v7 + 56));
      }
      v7 += 80LL;
      v6 += 10;
    }
    while ( v5 != v7 );
    v5 = *a1;
    v8 = (_QWORD *)(*a1 + 80LL * *((unsigned int *)a1 + 2));
    if ( v8 != (_QWORD *)*a1 )
    {
      do
      {
        v8 -= 10;
        v9 = (_QWORD *)v8[6];
        if ( v9 != v8 + 8 )
          j_j___libc_free_0(v9, v8[8] + 1LL);
        v10 = (_QWORD *)v8[2];
        if ( v10 != v8 + 4 )
          j_j___libc_free_0(v10, v8[4] + 1LL);
      }
      while ( v8 != (_QWORD *)v5 );
      v5 = *a1;
    }
  }
  if ( (unsigned __int64 *)v5 != a1 + 2 )
    _libc_free(v5);
  *((_DWORD *)a1 + 3) = v4;
  *a1 = v12;
  return v12;
}
