// Function: sub_253BF50
// Address: 0x253bf50
//
void __fastcall sub_253BF50(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // r12
  unsigned __int64 v8; // rdi
  __int64 v9; // rdx
  _QWORD *v10; // r15
  _QWORD *v11; // r12
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v15; // rbx
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rdi

  v2 = *(_QWORD *)(a1 + 264);
  *(_QWORD *)a1 = &unk_4A17418;
  while ( v2 )
  {
    sub_253B2D0(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3);
  }
  v4 = *(_QWORD *)(a1 + 200);
  if ( v4 != a1 + 216 )
    _libc_free(v4);
  v5 = *(unsigned int *)(a1 + 192);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD **)(a1 + 176);
    v7 = &v6[9 * v5];
    do
    {
      if ( *v6 != -8192 && *v6 != -4096 )
      {
        v8 = v6[1];
        if ( (_QWORD *)v8 != v6 + 3 )
          _libc_free(v8);
      }
      v6 += 9;
    }
    while ( v7 != v6 );
    v5 = *(unsigned int *)(a1 + 192);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 176), 72 * v5, 8);
  v9 = *(unsigned int *)(a1 + 160);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD **)(a1 + 144);
    v11 = &v10[12 * v9];
    while ( 1 )
    {
      if ( *v10 == 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v10[1] != 0x7FFFFFFFFFFFFFFFLL )
          goto LABEL_17;
      }
      else if ( *v10 != 0x7FFFFFFFFFFFFFFELL || v10[1] != 0x7FFFFFFFFFFFFFFELL )
      {
LABEL_17:
        v12 = v10[8];
        while ( v12 )
        {
          sub_253AEE0(*(_QWORD *)(v12 + 24));
          v13 = v12;
          v12 = *(_QWORD *)(v12 + 16);
          j_j___libc_free_0(v13);
        }
        v14 = v10[2];
        if ( (_QWORD *)v14 != v10 + 4 )
          _libc_free(v14);
      }
      v10 += 12;
      if ( v11 == v10 )
      {
        v9 = *(unsigned int *)(a1 + 160);
        break;
      }
    }
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 144), 96 * v9, 8);
  v15 = *(_QWORD *)(a1 + 8);
  v16 = v15 + 112LL * *(unsigned int *)(a1 + 16);
  if ( v15 != v16 )
  {
    do
    {
      v16 -= 112LL;
      v17 = *(_QWORD *)(v16 + 32);
      if ( v17 != v16 + 48 )
        _libc_free(v17);
    }
    while ( v15 != v16 );
    v16 = *(_QWORD *)(a1 + 8);
  }
  if ( v16 != a1 + 24 )
    _libc_free(v16);
}
