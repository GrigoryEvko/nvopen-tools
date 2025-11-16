// Function: sub_27F2AE0
// Address: 0x27f2ae0
//
__int64 __fastcall sub_27F2AE0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 result; // rax
  __int64 v8; // r13
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rbx
  unsigned __int64 v15; // rdi

  v6 = *a1;
  result = 15LL * *((unsigned int *)a1 + 2);
  v8 = *a1 + 120LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v8 )
  {
    do
    {
      if ( a2 )
      {
        *(_DWORD *)(a2 + 24) = 0;
        *(_QWORD *)(a2 + 8) = 0;
        *(_DWORD *)(a2 + 16) = 0;
        *(_DWORD *)(a2 + 20) = 0;
        *(_QWORD *)a2 = 1;
        v10 = *(_QWORD *)(v6 + 8);
        ++*(_QWORD *)v6;
        v11 = *(_QWORD *)(a2 + 8);
        *(_QWORD *)(a2 + 8) = v10;
        LODWORD(v10) = *(_DWORD *)(v6 + 16);
        *(_QWORD *)(v6 + 8) = v11;
        LODWORD(v11) = *(_DWORD *)(a2 + 16);
        *(_DWORD *)(a2 + 16) = v10;
        LODWORD(v10) = *(_DWORD *)(v6 + 20);
        *(_DWORD *)(v6 + 16) = v11;
        LODWORD(v11) = *(_DWORD *)(a2 + 20);
        *(_DWORD *)(a2 + 20) = v10;
        v12 = *(unsigned int *)(v6 + 24);
        *(_DWORD *)(v6 + 20) = v11;
        LODWORD(v11) = *(_DWORD *)(a2 + 24);
        *(_DWORD *)(a2 + 24) = v12;
        *(_DWORD *)(v6 + 24) = v11;
        *(_QWORD *)(a2 + 32) = a2 + 48;
        *(_DWORD *)(a2 + 40) = 0;
        *(_DWORD *)(a2 + 44) = 8;
        if ( *(_DWORD *)(v6 + 40) )
          sub_27EBCF0(a2 + 32, (char **)(v6 + 32), v12, a4, a5, a6);
        *(_BYTE *)(a2 + 112) = *(_BYTE *)(v6 + 112);
      }
      v6 += 120;
      a2 += 120;
    }
    while ( v8 != v6 );
    v13 = *a1;
    result = 15LL * *((unsigned int *)a1 + 2);
    v14 = *a1 + 120LL * *((unsigned int *)a1 + 2);
    if ( v14 != *a1 )
    {
      do
      {
        v14 -= 120;
        v15 = *(_QWORD *)(v14 + 32);
        if ( v15 != v14 + 48 )
          _libc_free(v15);
        result = sub_C7D6A0(*(_QWORD *)(v14 + 8), 8LL * *(unsigned int *)(v14 + 24), 8);
      }
      while ( v14 != v13 );
    }
  }
  return result;
}
