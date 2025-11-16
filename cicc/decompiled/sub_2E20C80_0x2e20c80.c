// Function: sub_2E20C80
// Address: 0x2e20c80
//
void __fastcall sub_2E20C80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // r9
  unsigned int v11; // r13d
  _QWORD *v12; // rax
  _QWORD *v13; // rsi
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // r14
  __int64 v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi

  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)a1 = v7;
  *(_QWORD *)(a1 + 16) = a4;
  v11 = *(_DWORD *)(v7 + 44);
  if ( v11 != *(_DWORD *)(a1 + 40) )
  {
    v12 = (_QWORD *)sub_2207820(176LL * v11 + 8);
    v8 = (__int64)v12;
    if ( v12 )
    {
      *v12 = v11;
      v13 = v12 + 1;
      if ( v11 )
      {
        v14 = v12 + 1;
        v8 += 176LL * v11 + 8;
        do
        {
          *v14 = 0;
          v14[4] = v14 + 6;
          v15 = v14 + 16;
          v14 += 22;
          *(v14 - 21) = 0;
          *(v14 - 19) = 0;
          *((_DWORD *)v14 - 34) = 0;
          *((_DWORD *)v14 - 33) = 4;
          *(v14 - 8) = v15;
          *((_DWORD *)v14 - 14) = 0;
          *((_DWORD *)v14 - 13) = 4;
          *((_BYTE *)v14 - 16) = 0;
          *((_BYTE *)v14 - 15) = 0;
          *((_DWORD *)v14 - 3) = 0;
          *((_DWORD *)v14 - 2) = 0;
        }
        while ( v14 != (_QWORD *)v8 );
      }
    }
    else
    {
      v13 = 0;
    }
    v16 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(a1 + 56) = v13;
    if ( v16 )
    {
      v17 = v16 + 176LL * *(_QWORD *)(v16 - 8);
      while ( v16 != v17 )
      {
        v17 -= 176;
        v18 = *(_QWORD *)(v17 + 112);
        if ( v18 != v17 + 128 )
          _libc_free(v18);
        v19 = *(_QWORD *)(v17 + 32);
        if ( v19 != v17 + 48 )
          _libc_free(v19);
      }
      j_j_j___libc_free_0_0(v16 - 8);
    }
  }
  sub_2E1A0B0(a1 + 40, *(_QWORD *)(a1 + 32), v11, v8, v9, v10);
  ++*(_DWORD *)(a1 + 24);
}
