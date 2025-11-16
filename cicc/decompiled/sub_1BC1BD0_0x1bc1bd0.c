// Function: sub_1BC1BD0
// Address: 0x1bc1bd0
//
void __fastcall sub_1BC1BD0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // r14
  unsigned __int64 *v10; // rax
  unsigned __int64 *v11; // r12
  __int64 v12; // r15
  __int64 v13; // rdx
  unsigned __int64 *v14; // r15
  unsigned __int64 *v15; // [rsp+8h] [rbp-38h]

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
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
  v9 = malloc(144 * v4);
  if ( !v9 )
    sub_16BD1C0("Allocation failed", 1u);
  v10 = *(unsigned __int64 **)a1;
  v11 = (unsigned __int64 *)(*(_QWORD *)a1 + 144LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v11 )
  {
    v12 = v9;
    do
    {
      if ( v12 )
      {
        *(_DWORD *)(v12 + 8) = 0;
        *(_QWORD *)v12 = v12 + 16;
        *(_DWORD *)(v12 + 12) = 16;
        v13 = *((unsigned int *)v10 + 2);
        if ( (_DWORD)v13 )
        {
          v15 = v10;
          sub_1BB9A40(v12, (char **)v10, v13, v6, v7, v8);
          v10 = v15;
        }
      }
      v10 += 18;
      v12 += 144;
    }
    while ( v11 != v10 );
    v14 = *(unsigned __int64 **)a1;
    v11 = (unsigned __int64 *)(*(_QWORD *)a1 + 144LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v11 )
    {
      do
      {
        v11 -= 18;
        if ( (unsigned __int64 *)*v11 != v11 + 2 )
          _libc_free(*v11);
      }
      while ( v11 != v14 );
      v11 = *(unsigned __int64 **)a1;
    }
  }
  if ( v11 != (unsigned __int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v11);
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 12) = v4;
}
