// Function: sub_A552A0
// Address: 0xa552a0
//
void __fastcall sub_A552A0(__int64 a1, __int64 a2)
{
  int v3; // ecx
  __int64 v4; // r8
  __int64 v5; // r13
  __int64 v6; // r13
  __int64 v7; // rbx
  _QWORD *v8; // rdi
  __int64 v9; // r13
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // rbx
  _QWORD *v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // r13
  __int64 v16; // r8
  __int64 v17; // r13
  __int64 v18; // rbx
  _QWORD *v19; // rdi
  void (__fastcall *v20)(__int64, __int64, __int64); // rax
  void (__fastcall *v21)(__int64, __int64, __int64); // rax

  v3 = *(_DWORD *)(a1 + 380);
  v4 = *(_QWORD *)(a1 + 368);
  *(_QWORD *)a1 = &unk_49D9A20;
  if ( v3 )
  {
    v5 = *(unsigned int *)(a1 + 376);
    if ( (_DWORD)v5 )
    {
      v6 = 8 * v5;
      v7 = 0;
      do
      {
        v8 = *(_QWORD **)(v4 + v7);
        if ( v8 != (_QWORD *)-8LL && v8 )
        {
          a2 = *v8 + 17LL;
          sub_C7D6A0(v8, a2, 8);
          v4 = *(_QWORD *)(a1 + 368);
        }
        v7 += 8;
      }
      while ( v6 != v7 );
    }
  }
  _libc_free(v4, a2);
  if ( *(_DWORD *)(a1 + 348) )
  {
    v9 = *(unsigned int *)(a1 + 344);
    v10 = *(_QWORD *)(a1 + 336);
    if ( (_DWORD)v9 )
    {
      v11 = 8 * v9;
      v12 = 0;
      do
      {
        v13 = *(_QWORD **)(v10 + v12);
        if ( v13 != (_QWORD *)-8LL && v13 )
        {
          a2 = *v13 + 17LL;
          sub_C7D6A0(v13, a2, 8);
          v10 = *(_QWORD *)(a1 + 336);
        }
        v12 += 8;
      }
      while ( v11 != v12 );
    }
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 336);
  }
  _libc_free(v10, a2);
  v14 = 16LL * *(unsigned int *)(a1 + 320);
  sub_C7D6A0(*(_QWORD *)(a1 + 304), v14, 8);
  if ( *(_DWORD *)(a1 + 276) )
  {
    v15 = *(unsigned int *)(a1 + 272);
    v16 = *(_QWORD *)(a1 + 264);
    if ( (_DWORD)v15 )
    {
      v17 = 8 * v15;
      v18 = 0;
      do
      {
        v19 = *(_QWORD **)(v16 + v18);
        if ( v19 != (_QWORD *)-8LL && v19 )
        {
          v14 = *v19 + 17LL;
          sub_C7D6A0(v19, v14, 8);
          v16 = *(_QWORD *)(a1 + 264);
        }
        v18 += 8;
      }
      while ( v17 != v18 );
    }
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 264);
  }
  _libc_free(v16, v14);
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 16LL * *(unsigned int *)(a1 + 248), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 192), 16LL * *(unsigned int *)(a1 + 208), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 152), 16LL * *(unsigned int *)(a1 + 168), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 112), 16LL * *(unsigned int *)(a1 + 128), 8);
  v20 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 80);
  if ( v20 )
    v20(a1 + 64, a1 + 64, 3);
  v21 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 48);
  if ( v21 )
    v21(a1 + 32, a1 + 32, 3);
  nullsub_38();
}
