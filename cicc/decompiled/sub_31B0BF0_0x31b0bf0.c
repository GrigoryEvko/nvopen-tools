// Function: sub_31B0BF0
// Address: 0x31b0bf0
//
void __fastcall sub_31B0BF0(unsigned __int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rsi
  _QWORD *v6; // r12
  _QWORD *v7; // rbx
  unsigned __int64 v8; // r14
  __int64 *v9; // rax
  unsigned __int64 v10; // rdi
  __int64 v11; // rdx
  unsigned __int64 v12; // rdi

  v2 = *(_QWORD *)(a1 + 272);
  v3 = v2 + 8LL * *(unsigned int *)(a1 + 280);
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 - 8);
      v3 -= 8LL;
      if ( v4 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 272);
  }
  if ( v3 != a1 + 288 )
    _libc_free(v3);
  if ( *(_BYTE *)(a1 + 264) )
    sub_3187270(*(_QWORD *)(a1 + 168), *(_QWORD *)(a1 + 256));
  v5 = *(unsigned int *)(a1 + 240);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD **)(a1 + 224);
    v7 = &v6[2 * v5];
    do
    {
      if ( *v6 != -8192 && *v6 != -4096 )
      {
        v8 = v6[1];
        if ( v8 )
        {
          v9 = *(__int64 **)v8;
          v10 = *(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8);
          if ( *(_QWORD *)v8 != v10 )
          {
            do
            {
              v11 = *v9++;
              *(_QWORD *)(v11 + 32) = 0;
            }
            while ( (__int64 *)v10 != v9 );
            v10 = *(_QWORD *)v8;
          }
          if ( v10 != v8 + 16 )
            _libc_free(v10);
          j_j___libc_free_0(v8);
        }
      }
      v6 += 2;
    }
    while ( v7 != v6 );
    v5 = *(unsigned int *)(a1 + 240);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 224), 16 * v5, 8);
  sub_31B0A30(a1 + 40);
  v12 = *(_QWORD *)(a1 + 8);
  if ( v12 )
    j_j___libc_free_0(v12);
  j_j___libc_free_0(a1);
}
