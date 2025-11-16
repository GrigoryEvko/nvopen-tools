// Function: sub_29F7BD0
// Address: 0x29f7bd0
//
void __fastcall sub_29F7BD0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  _QWORD *v10; // r14
  unsigned __int64 *v11; // rax
  __int64 v12; // rcx
  unsigned __int64 *v13; // r12
  _QWORD *v14; // rbx
  unsigned __int64 *v15; // rbx
  int v16; // ebx
  unsigned __int64 *v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x98u, v18, a6);
  v9 = *(unsigned int *)(a1 + 8);
  v10 = (_QWORD *)v6;
  v11 = *(unsigned __int64 **)a1;
  v12 = 9 * v9;
  v13 = (unsigned __int64 *)(*(_QWORD *)a1 + 152 * v9);
  if ( *(unsigned __int64 **)a1 != v13 )
  {
    v14 = v10;
    do
    {
      if ( v14 )
      {
        v14[1] = 0;
        *v14 = v14 + 3;
        v14[2] = 128;
        if ( v11[1] )
        {
          v17 = v11;
          sub_29F3DD0((__int64)v14, (char **)v11, (__int64)(v14 + 3), v12, v7, v8);
          v11 = v17;
        }
      }
      v11 += 19;
      v14 += 19;
    }
    while ( v13 != v11 );
    v15 = *(unsigned __int64 **)a1;
    v13 = (unsigned __int64 *)(*(_QWORD *)a1 + 152LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v13 )
    {
      do
      {
        v13 -= 19;
        if ( (unsigned __int64 *)*v13 != v13 + 3 )
          _libc_free(*v13);
      }
      while ( v13 != v15 );
      v13 = *(unsigned __int64 **)a1;
    }
  }
  v16 = v18[0];
  if ( (unsigned __int64 *)(a1 + 16) != v13 )
    _libc_free((unsigned __int64)v13);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v16;
}
