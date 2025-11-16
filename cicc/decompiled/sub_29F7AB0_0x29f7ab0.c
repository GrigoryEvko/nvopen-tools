// Function: sub_29F7AB0
// Address: 0x29f7ab0
//
void __fastcall sub_29F7AB0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // r15
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  _QWORD *v12; // r14
  unsigned __int64 *v13; // rax
  __int64 v14; // rcx
  unsigned __int64 *v15; // r12
  _QWORD *v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rdi
  unsigned __int64 *v19; // rbx
  int v20; // ebx
  unsigned __int64 *v21; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (unsigned __int64 *)(a1 + 16);
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x58u, v22, a6);
  v11 = *(unsigned int *)(a1 + 8);
  v12 = (_QWORD *)v8;
  v13 = *(unsigned __int64 **)a1;
  v14 = 5 * v11;
  v15 = (unsigned __int64 *)(*(_QWORD *)a1 + 88 * v11);
  if ( *(unsigned __int64 **)a1 != v15 )
  {
    v16 = v12;
    do
    {
      while ( 1 )
      {
        if ( v16 )
        {
          v17 = (__int64)(v16 + 3);
          v16[1] = 0;
          *v16 = v16 + 3;
          v16[2] = 64;
          if ( v13[1] )
            break;
        }
        v13 += 11;
        v16 += 11;
        if ( v15 == v13 )
          goto LABEL_7;
      }
      v18 = (__int64)v16;
      v21 = v13;
      v16 += 11;
      sub_29F3DD0(v18, (char **)v13, v17, v14, v9, v10);
      v13 = v21 + 11;
    }
    while ( v15 != v21 + 11 );
LABEL_7:
    v19 = *(unsigned __int64 **)a1;
    v15 = (unsigned __int64 *)(*(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v15 )
    {
      do
      {
        v15 -= 11;
        if ( (unsigned __int64 *)*v15 != v15 + 3 )
          _libc_free(*v15);
      }
      while ( v15 != v19 );
      v15 = *(unsigned __int64 **)a1;
    }
  }
  v20 = v22[0];
  if ( v6 != v15 )
    _libc_free((unsigned __int64)v15);
  *(_QWORD *)a1 = v12;
  *(_DWORD *)(a1 + 12) = v20;
}
