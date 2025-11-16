// Function: sub_18E88A0
// Address: 0x18e88a0
//
void __fastcall sub_18E88A0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // r15
  __int64 v10; // rdx
  unsigned __int64 *v11; // rax
  __int64 v12; // rcx
  unsigned __int64 *v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rdx
  unsigned __int64 *v16; // rbx
  unsigned __int64 *v17; // [rsp+8h] [rbp-38h]

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
  v6 = v2;
  if ( v2 > 0xFFFFFFFF )
    v6 = 0xFFFFFFFFLL;
  v9 = malloc(152 * v6);
  if ( !v9 )
    sub_16BD1C0("Allocation failed", 1u);
  v10 = *(unsigned int *)(a1 + 8);
  v11 = *(unsigned __int64 **)a1;
  v12 = 9 * v10;
  v13 = (unsigned __int64 *)(*(_QWORD *)a1 + 152 * v10);
  if ( *(unsigned __int64 **)a1 != v13 )
  {
    v14 = v9;
    do
    {
      if ( v14 )
      {
        *(_DWORD *)(v14 + 8) = 0;
        *(_QWORD *)v14 = v14 + 16;
        *(_DWORD *)(v14 + 12) = 8;
        v15 = *((unsigned int *)v11 + 2);
        if ( (_DWORD)v15 )
        {
          v17 = v11;
          sub_18E63F0(v14, (char **)v11, v15, v12, v7, v8);
          v11 = v17;
        }
        *(_QWORD *)(v14 + 144) = v11[18];
      }
      v11 += 19;
      v14 += 152;
    }
    while ( v13 != v11 );
    v16 = *(unsigned __int64 **)a1;
    v13 = (unsigned __int64 *)(*(_QWORD *)a1 + 152LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v13 )
    {
      do
      {
        v13 -= 19;
        if ( (unsigned __int64 *)*v13 != v13 + 2 )
          _libc_free(*v13);
      }
      while ( v13 != v16 );
      v13 = *(unsigned __int64 **)a1;
    }
  }
  if ( v13 != (unsigned __int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v13);
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 12) = v6;
}
