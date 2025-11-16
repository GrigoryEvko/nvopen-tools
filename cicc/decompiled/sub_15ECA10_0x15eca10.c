// Function: sub_15ECA10
// Address: 0x15eca10
//
__int64 __fastcall sub_15ECA10(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // r13
  __int64 v9; // r12
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rdx
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // r15
  _QWORD *v17; // r14
  unsigned __int64 v18; // rbx
  _QWORD *v19; // r12
  int v21; // [rsp+0h] [rbp-50h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23; // [rsp+18h] [rbp-38h]

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  v2 = (((((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
         | (*(unsigned int *)(a1 + 12) + 2LL)
         | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | (((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v3 = a2;
  v4 = (v2 | (v2 >> 16) | HIDWORD(v2)) + 1;
  v5 = 0xFFFFFFFFLL;
  if ( v4 >= a2 )
    v3 = v4;
  if ( v3 <= 0xFFFFFFFF )
    v5 = v3;
  v21 = v5;
  v22 = malloc(192 * v5);
  if ( !v22 )
    sub_16BD1C0("Allocation failed");
  v7 = *(_QWORD *)a1;
  v8 = *(_QWORD *)a1 + 192LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = v22;
    do
    {
      if ( v9 )
      {
        *(_DWORD *)v9 = *(_DWORD *)v7;
        *(_DWORD *)(v9 + 4) = *(_DWORD *)(v7 + 4);
        *(_BYTE *)(v9 + 8) = *(_BYTE *)(v7 + 8);
        *(_BYTE *)(v9 + 9) = *(_BYTE *)(v7 + 9);
        *(_BYTE *)(v9 + 10) = *(_BYTE *)(v7 + 10);
        *(_BYTE *)(v9 + 11) = *(_BYTE *)(v7 + 11);
        v10 = *(_DWORD *)(v7 + 12);
        *(_DWORD *)(v9 + 24) = 0;
        *(_DWORD *)(v9 + 12) = v10;
        *(_QWORD *)(v9 + 16) = v9 + 32;
        *(_DWORD *)(v9 + 28) = 1;
        v11 = *(unsigned int *)(v7 + 24);
        if ( (_DWORD)v11 )
          sub_15EB170(v9 + 16, v7 + 16, v11, v6);
        *(_DWORD *)(v9 + 72) = 0;
        *(_QWORD *)(v9 + 64) = v9 + 80;
        *(_DWORD *)(v9 + 76) = 2;
        if ( *(_DWORD *)(v7 + 72) )
          sub_15EC260(v9 + 64, v7 + 64);
      }
      v7 += 192LL;
      v9 += 192;
    }
    while ( v8 != v7 );
    v23 = *(_QWORD *)a1;
    v8 = *(_QWORD *)a1 + 192LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v12 = *(unsigned int *)(v8 - 120);
        v13 = *(_QWORD *)(v8 - 128);
        v8 -= 192LL;
        v14 = v13 + 56 * v12;
        if ( v13 != v14 )
        {
          do
          {
            v15 = *(unsigned int *)(v14 - 40);
            v16 = *(_QWORD *)(v14 - 48);
            v14 -= 56LL;
            v15 *= 32;
            v17 = (_QWORD *)(v16 + v15);
            if ( v16 != v16 + v15 )
            {
              do
              {
                v17 -= 4;
                if ( (_QWORD *)*v17 != v17 + 2 )
                  j_j___libc_free_0(*v17, v17[2] + 1LL);
              }
              while ( (_QWORD *)v16 != v17 );
              v16 = *(_QWORD *)(v14 + 8);
            }
            if ( v16 != v14 + 24 )
              _libc_free(v16);
          }
          while ( v13 != v14 );
          v13 = *(_QWORD *)(v8 + 64);
        }
        if ( v13 != v8 + 80 )
          _libc_free(v13);
        v18 = *(_QWORD *)(v8 + 16);
        v19 = (_QWORD *)(v18 + 32LL * *(unsigned int *)(v8 + 24));
        if ( (_QWORD *)v18 != v19 )
        {
          do
          {
            v19 -= 4;
            if ( (_QWORD *)*v19 != v19 + 2 )
              j_j___libc_free_0(*v19, v19[2] + 1LL);
          }
          while ( (_QWORD *)v18 != v19 );
          v18 = *(_QWORD *)(v8 + 16);
        }
        if ( v18 != v8 + 32 )
          _libc_free(v18);
      }
      while ( v8 != v23 );
      v8 = *(_QWORD *)a1;
    }
  }
  if ( v8 != a1 + 16 )
    _libc_free(v8);
  *(_QWORD *)a1 = v22;
  *(_DWORD *)(a1 + 12) = v21;
  return a1;
}
