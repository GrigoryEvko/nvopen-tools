// Function: sub_1F232A0
// Address: 0x1f232a0
//
__int64 __fastcall sub_1F232A0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rcx
  unsigned __int64 v9; // r12
  _QWORD *v10; // rax
  _QWORD *v11; // rcx
  __int64 v12; // rbx
  unsigned __int64 *v13; // r13
  unsigned __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  int v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]

  v2 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v4 = ((v3
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v3
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v5 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v5 >= a2 )
    v2 = v5;
  v6 = 0xFFFFFFFFLL;
  if ( v2 <= 0xFFFFFFFF )
    v6 = v2;
  v19 = v6;
  v20 = malloc(8 * v6);
  if ( !v20 )
    sub_16BD1C0("Allocation failed", 1u);
  v7 = *(_QWORD **)a1;
  v8 = 8LL * *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1 + v8;
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = (_QWORD *)v20;
    v11 = (_QWORD *)(v20 + v8);
    do
    {
      if ( v10 )
      {
        *v10 = *v7;
        *v7 = 0;
      }
      ++v10;
      ++v7;
    }
    while ( v10 != v11 );
    v9 = *(_QWORD *)a1;
    v12 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( v12 != *(_QWORD *)a1 )
    {
      do
      {
        v13 = *(unsigned __int64 **)(v12 - 8);
        v12 -= 8;
        if ( v13 )
        {
          sub_1DB4CE0((__int64)v13);
          v14 = v13[12];
          if ( v14 )
          {
            v15 = *(_QWORD *)(v14 + 16);
            while ( v15 )
            {
              sub_1F21070(*(_QWORD *)(v15 + 24));
              v16 = v15;
              v15 = *(_QWORD *)(v15 + 16);
              j_j___libc_free_0(v16, 56);
            }
            j_j___libc_free_0(v14, 48);
          }
          v17 = v13[8];
          if ( (unsigned __int64 *)v17 != v13 + 10 )
            _libc_free(v17);
          if ( (unsigned __int64 *)*v13 != v13 + 2 )
            _libc_free(*v13);
          j_j___libc_free_0(v13, 120);
        }
      }
      while ( v9 != v12 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_QWORD *)a1 = v20;
  *(_DWORD *)(a1 + 12) = v19;
  return a1;
}
