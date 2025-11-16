// Function: sub_1B679A0
// Address: 0x1b679a0
//
void __fastcall sub_1B679A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v6; // rax
  int v7; // eax
  size_t **v8; // rax
  size_t *v9; // r12

  v4 = *(_QWORD *)(a2 + 48);
  if ( v4 )
  {
    v6 = sub_1633B90(a1, *(void **)a4, *(_QWORD *)(a4 + 8));
    *(_DWORD *)(v6 + 8) = *(_DWORD *)(v4 + 8);
    *(_QWORD *)(a2 + 48) = v6;
    v7 = sub_16D1B30((__int64 *)(a1 + 128), *(unsigned __int8 **)a3, *(_QWORD *)(a3 + 8));
    if ( v7 == -1 )
      v8 = (size_t **)(*(_QWORD *)(a1 + 128) + 8LL * *(unsigned int *)(a1 + 136));
    else
      v8 = (size_t **)(*(_QWORD *)(a1 + 128) + 8LL * v7);
    v9 = *v8;
    sub_16D1CB0(a1 + 128, *v8);
    _libc_free((unsigned __int64)v9);
  }
}
