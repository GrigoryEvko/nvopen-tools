// Function: sub_18E8DE0
// Address: 0x18e8de0
//
__int64 __fastcall sub_18E8DE0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  int v8; // r9d
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rdx
  unsigned __int64 v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // r15
  unsigned __int64 *v17; // rbx
  unsigned int v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20; // [rsp+10h] [rbp-40h]
  __int64 v21; // [rsp+18h] [rbp-38h]

  v2 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = (((((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
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
  v4 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  v5 = 0xFFFFFFFFLL;
  if ( v4 >= a2 )
    v2 = v4;
  if ( v2 <= 0xFFFFFFFF )
    v5 = v2;
  v19 = v5;
  v21 = malloc(632 * v5);
  if ( !v21 )
    sub_16BD1C0("Allocation failed", 1u);
  v9 = *(_QWORD *)a1;
  v10 = *(_QWORD *)a1 + 632LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = v21;
    do
    {
      if ( v11 )
      {
        v12 = *(_QWORD *)v9;
        *(_DWORD *)(v11 + 16) = 0;
        *(_DWORD *)(v11 + 20) = 4;
        *(_QWORD *)v11 = v12;
        *(_QWORD *)(v11 + 8) = v11 + 24;
        v13 = *(unsigned int *)(v9 + 16);
        if ( (_DWORD)v13 )
        {
          v20 = v9;
          sub_18E8A60(v11 + 8, v9 + 8, v13, v6, v7, v8);
          v9 = v20;
        }
      }
      v9 += 632LL;
      v11 += 632;
    }
    while ( v10 != v9 );
    v14 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a1 + 632LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v15 = *(unsigned int *)(v10 - 616);
        v16 = *(_QWORD *)(v10 - 624);
        v10 -= 632LL;
        v17 = (unsigned __int64 *)(v16 + 152 * v15);
        if ( (unsigned __int64 *)v16 != v17 )
        {
          do
          {
            v17 -= 19;
            if ( (unsigned __int64 *)*v17 != v17 + 2 )
              _libc_free(*v17);
          }
          while ( (unsigned __int64 *)v16 != v17 );
          v16 = *(_QWORD *)(v10 + 8);
        }
        if ( v16 != v10 + 24 )
          _libc_free(v16);
      }
      while ( v10 != v14 );
      v10 = *(_QWORD *)a1;
    }
  }
  if ( v10 != a1 + 16 )
    _libc_free(v10);
  *(_QWORD *)a1 = v21;
  *(_DWORD *)(a1 + 12) = v19;
  return v19;
}
