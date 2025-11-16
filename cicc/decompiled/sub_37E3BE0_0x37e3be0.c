// Function: sub_37E3BE0
// Address: 0x37e3be0
//
__int64 __fastcall sub_37E3BE0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // rax
  unsigned __int64 v9; // r15
  _DWORD *v10; // rbx
  _DWORD *v11; // r13
  _DWORD *v12; // rdi
  _DWORD *v13; // rax
  _DWORD *v14; // rbx
  int v15; // ebx
  __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x50u, v18, a6);
  v17 = v8;
  v9 = *(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = (_DWORD *)(v8 + 80);
    v11 = *(_DWORD **)a1;
    do
    {
      v12 = v10 - 20;
      if ( v10 != (_DWORD *)80 )
      {
        *((_QWORD *)v10 - 10) = 0;
        v13 = v10 - 16;
        v12[2] = 1;
        *(v10 - 17) = 0;
        do
        {
          if ( v13 )
            *v13 = -1;
          v13 += 4;
        }
        while ( v13 != v10 );
        sub_37E3AB0(v12, v11);
      }
      v11 += 20;
      v10 += 20;
    }
    while ( (_DWORD *)v9 != v11 );
    v14 = *(_DWORD **)a1;
    v9 = *(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v9 -= 80LL;
        if ( (*(_BYTE *)(v9 + 8) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(v9 + 16), 16LL * *(unsigned int *)(v9 + 24), 8);
      }
      while ( (_DWORD *)v9 != v14 );
      v9 = *(_QWORD *)a1;
    }
  }
  v15 = v18[0];
  if ( v6 != v9 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v15;
  *(_QWORD *)a1 = v17;
  return v17;
}
