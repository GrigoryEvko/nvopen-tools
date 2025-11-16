// Function: sub_2B48360
// Address: 0x2b48360
//
void __fastcall sub_2B48360(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // r13
  __int64 v18; // r12
  __int64 v19; // rax
  unsigned __int64 v20; // r14
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // rdi

  v6 = *a1;
  v7 = *a1 + 112LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v7 )
  {
    do
    {
      while ( 1 )
      {
        v10 = 112;
        if ( a2 )
        {
          *(_DWORD *)(a2 + 8) = 0;
          *(_QWORD *)a2 = a2 + 16;
          *(_DWORD *)(a2 + 12) = 6;
          v14 = *(unsigned int *)(v6 + 8);
          if ( (_DWORD)v14 )
            sub_2B09BF0(a2, (char **)v6, v14, a4, a5, a6);
          *(_DWORD *)(a2 + 88) = 0;
          v10 = a2 + 112;
          *(_QWORD *)(a2 + 72) = 0;
          *(_DWORD *)(a2 + 80) = 0;
          *(_DWORD *)(a2 + 84) = 0;
          *(_QWORD *)(a2 + 64) = 1;
          v11 = *(_QWORD *)(v6 + 72);
          ++*(_QWORD *)(v6 + 64);
          v12 = *(_QWORD *)(a2 + 72);
          *(_QWORD *)(a2 + 72) = v11;
          LODWORD(v11) = *(_DWORD *)(v6 + 80);
          *(_QWORD *)(v6 + 72) = v12;
          LODWORD(v12) = *(_DWORD *)(a2 + 80);
          *(_DWORD *)(a2 + 80) = v11;
          LODWORD(v11) = *(_DWORD *)(v6 + 84);
          *(_DWORD *)(v6 + 80) = v12;
          LODWORD(v12) = *(_DWORD *)(a2 + 84);
          *(_DWORD *)(a2 + 84) = v11;
          v13 = *(unsigned int *)(v6 + 88);
          *(_DWORD *)(v6 + 84) = v12;
          LODWORD(v12) = *(_DWORD *)(a2 + 88);
          *(_DWORD *)(a2 + 88) = v13;
          *(_DWORD *)(v6 + 88) = v12;
          *(_QWORD *)(a2 + 96) = a2 + 112;
          *(_DWORD *)(a2 + 104) = 0;
          *(_DWORD *)(a2 + 108) = 0;
          if ( *(_DWORD *)(v6 + 104) )
            break;
        }
        v6 += 112;
        a2 = v10;
        if ( v7 == v6 )
          goto LABEL_9;
      }
      v15 = v6 + 96;
      v16 = a2 + 96;
      v6 += 112;
      a2 += 112;
      sub_2B47FE0(v16, v15, v13, a4, a5, a6);
    }
    while ( v7 != v6 );
LABEL_9:
    v17 = *a1;
    v18 = *a1 + 112LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v18 )
    {
      do
      {
        v19 = *(unsigned int *)(v18 - 8);
        v20 = *(_QWORD *)(v18 - 16);
        v18 -= 112;
        v21 = v20 + 72 * v19;
        if ( v20 != v21 )
        {
          do
          {
            v21 -= 72LL;
            v22 = *(_QWORD *)(v21 + 8);
            if ( v22 != v21 + 24 )
              _libc_free(v22);
          }
          while ( v20 != v21 );
          v20 = *(_QWORD *)(v18 + 96);
        }
        if ( v20 != v18 + 112 )
          _libc_free(v20);
        sub_C7D6A0(*(_QWORD *)(v18 + 72), 16LL * *(unsigned int *)(v18 + 88), 8);
        if ( *(_QWORD *)v18 != v18 + 16 )
          _libc_free(*(_QWORD *)v18);
      }
      while ( v18 != v17 );
    }
  }
}
