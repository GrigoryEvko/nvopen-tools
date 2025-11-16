// Function: sub_2A7FD40
// Address: 0x2a7fd40
//
void __fastcall sub_2A7FD40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  const void *v9; // r14
  size_t v10; // r12
  int v11; // eax
  int v12; // eax
  size_t **v13; // rax
  size_t *v14; // r12
  __int64 v15; // r13

  v4 = *(_QWORD *)(a2 + 48);
  if ( v4 )
  {
    v6 = sub_BAA410(a1, *(void **)a4, *(_QWORD *)(a4 + 8));
    *(_DWORD *)(v6 + 8) = *(_DWORD *)(v4 + 8);
    sub_B2F990(a2, v6, v7, v8);
    v9 = *(const void **)a3;
    v10 = *(_QWORD *)(a3 + 8);
    v11 = sub_C92610();
    v12 = sub_C92860((__int64 *)(a1 + 128), v9, v10, v11);
    if ( v12 == -1 )
      v13 = (size_t **)(*(_QWORD *)(a1 + 128) + 8LL * *(unsigned int *)(a1 + 136));
    else
      v13 = (size_t **)(*(_QWORD *)(a1 + 128) + 8LL * v12);
    v14 = *v13;
    sub_C929B0(a1 + 128, *v13);
    v15 = *v14 + 73;
    if ( !*((_BYTE *)v14 + 52) )
      _libc_free(v14[4]);
    sub_C7D6A0((__int64)v14, v15, 8);
  }
}
