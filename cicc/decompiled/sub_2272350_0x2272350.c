// Function: sub_2272350
// Address: 0x2272350
//
void __fastcall sub_2272350(__int64 a1)
{
  __int64 v2; // r14
  unsigned __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // r13
  unsigned __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r15

  v2 = *(_QWORD *)(a1 + 96);
  v3 = v2 + 32LL * *(unsigned int *)(a1 + 104);
  *(_QWORD *)a1 = &unk_4A085B0;
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(unsigned int *)(v3 - 8);
      v3 -= 32LL;
      if ( (_DWORD)v4 )
      {
        v5 = *(_QWORD *)(v3 + 8);
        v6 = v5 + 48 * v4;
        do
        {
          while ( *(_QWORD *)v5 == -1 || *(_QWORD *)v5 == -2 )
          {
            v5 += 48;
            if ( v6 == v5 )
              goto LABEL_8;
          }
          v7 = *(unsigned int *)(v5 + 40);
          v8 = *(_QWORD *)(v5 + 24);
          v5 += 48;
          sub_C7D6A0(v8, 32 * v7, 8);
        }
        while ( v6 != v5 );
      }
LABEL_8:
      sub_C7D6A0(*(_QWORD *)(v3 + 8), 48LL * *(unsigned int *)(v3 + 24), 8);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 96);
  }
  if ( v3 != a1 + 112 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 8LL * *(unsigned int *)(a1 + 88), 8);
  v9 = *(_QWORD *)(a1 + 16);
  v10 = v9 + 32LL * *(unsigned int *)(a1 + 24);
  if ( v9 != v10 )
  {
    do
    {
      v11 = *(unsigned int *)(v10 - 8);
      v10 -= 32LL;
      if ( (_DWORD)v11 )
      {
        v12 = *(_QWORD *)(v10 + 8);
        v13 = v12 + 72 * v11;
        do
        {
          if ( *(_QWORD *)v12 != -8192 && *(_QWORD *)v12 != -4096 )
          {
            sub_C7D6A0(*(_QWORD *)(v12 + 48), 24LL * *(unsigned int *)(v12 + 64), 8);
            sub_C7D6A0(*(_QWORD *)(v12 + 16), 24LL * *(unsigned int *)(v12 + 32), 8);
          }
          v12 += 72;
        }
        while ( v13 != v12 );
      }
      sub_C7D6A0(*(_QWORD *)(v10 + 8), 72LL * *(unsigned int *)(v10 + 24), 8);
    }
    while ( v9 != v10 );
    v10 = *(_QWORD *)(a1 + 16);
  }
  if ( v10 != a1 + 32 )
    _libc_free(v10);
}
