// Function: sub_13AE5E0
// Address: 0x13ae5e0
//
void __fastcall sub_13AE5E0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // r14
  unsigned __int64 *v13; // r8
  unsigned __int64 *v14; // r8
  unsigned __int64 *v15; // r8
  unsigned __int64 *v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+8h] [rbp-38h]
  __int64 v18; // [rsp+8h] [rbp-38h]

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  v2 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v3 = ((v2
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v2
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = a2;
  v5 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v5 >= a2 )
    v4 = v5;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v6 = malloc(48 * v4);
  if ( !v6 )
    sub_16BD1C0("Allocation failed");
  v7 = *(_QWORD *)a1;
  v8 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = v6;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = *(_QWORD *)v7;
        *(_QWORD *)(v9 + 8) = *(_QWORD *)(v7 + 8);
        *(_DWORD *)(v9 + 16) = *(_DWORD *)(v7 + 16);
        *(_QWORD *)(v9 + 24) = *(_QWORD *)(v7 + 24);
        v10 = *(_QWORD *)(v7 + 32);
        *(_QWORD *)(v7 + 24) = 1;
        *(_QWORD *)(v9 + 32) = v10;
        v11 = *(_QWORD *)(v7 + 40);
        *(_QWORD *)(v7 + 32) = 1;
        *(_QWORD *)(v9 + 40) = v11;
        *(_QWORD *)(v7 + 40) = 1;
      }
      v7 += 48LL;
      v9 += 48;
    }
    while ( v8 != v7 );
    v8 = *(_QWORD *)a1;
    v12 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
    if ( v12 != *(_QWORD *)a1 )
    {
      do
      {
        while ( 1 )
        {
          v15 = *(unsigned __int64 **)(v12 - 8);
          v12 -= 48;
          if ( ((unsigned __int8)v15 & 1) == 0 && v15 )
          {
            v16 = v15;
            _libc_free(*v15);
            j_j___libc_free_0(v16, 24);
          }
          v13 = *(unsigned __int64 **)(v12 + 32);
          if ( ((unsigned __int8)v13 & 1) == 0 && v13 )
          {
            v17 = *(_QWORD *)(v12 + 32);
            _libc_free(*v13);
            j_j___libc_free_0(v17, 24);
          }
          v14 = *(unsigned __int64 **)(v12 + 24);
          if ( ((unsigned __int8)v14 & 1) == 0 )
          {
            if ( v14 )
              break;
          }
          if ( v8 == v12 )
            goto LABEL_26;
        }
        v18 = *(_QWORD *)(v12 + 24);
        _libc_free(*v14);
        j_j___libc_free_0(v18, 24);
      }
      while ( v8 != v12 );
LABEL_26:
      v8 = *(_QWORD *)a1;
    }
  }
  if ( v8 != a1 + 16 )
    _libc_free(v8);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v4;
}
