// Function: sub_1995CB0
// Address: 0x1995cb0
//
void __fastcall sub_1995CB0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rcx
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // r15
  unsigned __int64 *v11; // rax
  unsigned __int64 *v12; // r12
  __int64 v13; // rbx
  __int64 v14; // rdx
  unsigned __int64 *v15; // rbx
  unsigned __int64 *v16; // [rsp+8h] [rbp-38h]

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
  v10 = malloc(48 * v6);
  if ( !v10 )
    sub_16BD1C0("Allocation failed", 1u);
  v11 = *(unsigned __int64 **)a1;
  v12 = (unsigned __int64 *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v12 )
  {
    v13 = v10;
    do
    {
      if ( v13 )
      {
        *(_DWORD *)(v13 + 8) = 0;
        *(_QWORD *)v13 = v13 + 16;
        *(_DWORD *)(v13 + 12) = 1;
        v14 = *((unsigned int *)v11 + 2);
        if ( (_DWORD)v14 )
        {
          v16 = v11;
          sub_19938C0(v13, (char **)v11, v14, v7, v8, v9);
          v11 = v16;
        }
        *(_QWORD *)(v13 + 40) = v11[5];
      }
      v11 += 6;
      v13 += 48;
    }
    while ( v12 != v11 );
    v15 = *(unsigned __int64 **)a1;
    v12 = (unsigned __int64 *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v12 )
    {
      do
      {
        v12 -= 6;
        if ( (unsigned __int64 *)*v12 != v12 + 2 )
          _libc_free(*v12);
      }
      while ( v12 != v15 );
      v12 = *(unsigned __int64 **)a1;
    }
  }
  if ( v12 != (unsigned __int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v12);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v6;
}
