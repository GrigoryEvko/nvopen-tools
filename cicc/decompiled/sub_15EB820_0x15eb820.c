// Function: sub_15EB820
// Address: 0x15eb820
//
__int64 __fastcall sub_15EB820(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rcx
  _DWORD *v8; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // r14
  int v11; // edx
  __int64 v12; // rdx
  __int64 v13; // rdi
  _DWORD *v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // r15
  _QWORD *v17; // rbx
  unsigned int v19; // [rsp+8h] [rbp-48h]
  _DWORD *v20; // [rsp+10h] [rbp-40h]
  __int64 v21; // [rsp+18h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  v4 = (((((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
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
  v5 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  v6 = 0xFFFFFFFFLL;
  if ( v5 >= a2 )
    v3 = v5;
  if ( v3 <= 0xFFFFFFFF )
    v6 = v3;
  v19 = v6;
  v21 = malloc(56 * v6);
  if ( !v21 )
    sub_16BD1C0("Allocation failed");
  v7 = *(unsigned int *)(a1 + 8);
  v8 = *(_DWORD **)a1;
  v9 = *(_QWORD *)a1 + 56 * v7;
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = v21;
    do
    {
      while ( 1 )
      {
        if ( v10 )
        {
          v11 = *v8;
          *(_DWORD *)(v10 + 16) = 0;
          *(_DWORD *)(v10 + 20) = 1;
          *(_DWORD *)v10 = v11;
          *(_QWORD *)(v10 + 8) = v10 + 24;
          v12 = (unsigned int)v8[4];
          if ( (_DWORD)v12 )
            break;
        }
        v8 += 14;
        v10 += 56;
        if ( (_DWORD *)v9 == v8 )
          goto LABEL_15;
      }
      v13 = v10 + 8;
      v20 = v8;
      v10 += 56;
      sub_15EB170(v13, (__int64)(v8 + 2), v12, v7);
      v8 = v20 + 14;
    }
    while ( (_DWORD *)v9 != v20 + 14 );
LABEL_15:
    v14 = *(_DWORD **)a1;
    v9 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v15 = *(unsigned int *)(v9 - 40);
        v16 = *(_QWORD *)(v9 - 48);
        v9 -= 56LL;
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
          v16 = *(_QWORD *)(v9 + 8);
        }
        if ( v16 != v9 + 24 )
          _libc_free(v16);
      }
      while ( (_DWORD *)v9 != v14 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_QWORD *)a1 = v21;
  *(_DWORD *)(a1 + 12) = v19;
  return v19;
}
