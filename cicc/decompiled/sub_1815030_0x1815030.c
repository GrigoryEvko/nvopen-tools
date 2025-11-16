// Function: sub_1815030
// Address: 0x1815030
//
void __fastcall sub_1815030(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // r12
  _QWORD *v10; // rbx
  _QWORD *v11; // r12
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // r13
  unsigned __int64 v16; // rdi

  v2 = *(unsigned int *)(a1 + 336);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 320);
    v4 = &v3[7 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[3];
        while ( v5 )
        {
          sub_1814E60(*(_QWORD *)(v5 + 24));
          v6 = v5;
          v5 = *(_QWORD *)(v5 + 16);
          j_j___libc_free_0(v6, 40);
        }
      }
      v3 += 7;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 320));
  j___libc_free_0(*(_QWORD *)(a1 + 288));
  v7 = *(_QWORD *)(a1 + 248);
  if ( v7 )
    j_j___libc_free_0(v7, *(_QWORD *)(a1 + 264) - v7);
  j___libc_free_0(*(_QWORD *)(a1 + 224));
  v8 = *(_QWORD *)(a1 + 192);
  if ( v8 )
    j_j___libc_free_0(v8, *(_QWORD *)(a1 + 208) - v8);
  j___libc_free_0(*(_QWORD *)(a1 + 168));
  j___libc_free_0(*(_QWORD *)(a1 + 136));
  v9 = *(unsigned int *)(a1 + 64);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD **)(a1 + 48);
    v11 = &v10[2 * v9];
    do
    {
      if ( *v10 != -16 && *v10 != -8 )
      {
        v12 = v10[1];
        if ( v12 )
        {
          v13 = *(_QWORD *)(v12 + 24);
          if ( v13 )
            j_j___libc_free_0(v13, *(_QWORD *)(v12 + 40) - v13);
          j_j___libc_free_0(v12, 56);
        }
      }
      v10 += 2;
    }
    while ( v11 != v10 );
  }
  v14 = *(_QWORD *)(a1 + 48);
  v15 = a1 + 32;
  j___libc_free_0(v14);
  v16 = *(_QWORD *)(v15 - 16);
  if ( v16 != v15 )
    _libc_free(v16);
}
