// Function: sub_2AE94D0
// Address: 0x2ae94d0
//
void __fastcall sub_2AE94D0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  _DWORD *v4; // r13
  __int64 v5; // r15
  __int64 v6; // r14
  _DWORD *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  _DWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _DWORD *v17; // r12
  __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi

  v2 = *a1 + 192LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v2 )
  {
    v3 = a2;
    v4 = (_DWORD *)*a1;
    v5 = a2 + 144;
    v6 = a2 + 48;
    do
    {
      if ( v3 )
      {
        *(_QWORD *)v3 = 0;
        v7 = (_DWORD *)(v3 + 16);
        *(_DWORD *)(v3 + 8) = 1;
        *(_DWORD *)(v3 + 12) = 0;
        do
        {
          if ( v7 )
            *v7 = -1;
          v7 += 2;
        }
        while ( (_DWORD *)v6 != v7 );
        sub_2AE9220((_DWORD *)v3, v4);
        *(_DWORD *)(v3 + 56) = 0;
        *(_QWORD *)(v3 + 48) = v3 + 64;
        *(_DWORD *)(v3 + 60) = 4;
        v11 = (unsigned int)v4[14];
        if ( (_DWORD)v11 )
          sub_2AA8020(v6, (__int64)(v4 + 12), v11, v8, v9, v10);
        *(_QWORD *)(v3 + 96) = 0;
        *(_DWORD *)(v3 + 104) = 1;
        v12 = (_DWORD *)(v3 + 112);
        *(_DWORD *)(v3 + 108) = 0;
        do
        {
          if ( v12 )
            *v12 = -1;
          v12 += 2;
        }
        while ( (_DWORD *)v5 != v12 );
        sub_2AE9220((_DWORD *)(v3 + 96), v4 + 24);
        *(_DWORD *)(v3 + 152) = 0;
        *(_QWORD *)(v3 + 144) = v3 + 160;
        *(_DWORD *)(v3 + 156) = 4;
        if ( v4[38] )
          sub_2AA8020(v5, (__int64)(v4 + 36), v13, v14, v15, v16);
      }
      v4 += 48;
      v3 += 192;
      v5 += 192;
      v6 += 192;
    }
    while ( (_DWORD *)v2 != v4 );
    v17 = (_DWORD *)*a1;
    v18 = *a1 + 192LL * *((unsigned int *)a1 + 2);
    while ( (_DWORD *)v18 != v17 )
    {
      while ( 1 )
      {
        v18 -= 192;
        v19 = *(_QWORD *)(v18 + 144);
        if ( v19 != v18 + 160 )
          _libc_free(v19);
        if ( (*(_BYTE *)(v18 + 104) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(v18 + 112), 8LL * *(unsigned int *)(v18 + 120), 4);
        v20 = *(_QWORD *)(v18 + 48);
        if ( v20 != v18 + 64 )
          _libc_free(v20);
        if ( (*(_BYTE *)(v18 + 8) & 1) != 0 )
          break;
        sub_C7D6A0(*(_QWORD *)(v18 + 16), 8LL * *(unsigned int *)(v18 + 24), 4);
        if ( (_DWORD *)v18 == v17 )
          return;
      }
    }
  }
}
