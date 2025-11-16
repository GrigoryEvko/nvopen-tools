// Function: sub_E5BA60
// Address: 0xe5ba60
//
void __fastcall sub_E5BA60(__int64 a1, __int64 a2)
{
  bool v3; // zf
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rdi
  void (*v7)(void); // rax
  __int64 v8; // rdi
  void (*v9)(void); // rax
  __int64 v10; // rdi

  ++*(_QWORD *)(a1 + 80);
  v3 = *(_BYTE *)(a1 + 108) == 0;
  *(_BYTE *)(a1 + 33) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  if ( v3 )
  {
    v4 = 4 * (*(_DWORD *)(a1 + 100) - *(_DWORD *)(a1 + 104));
    v5 = *(unsigned int *)(a1 + 96);
    if ( v4 < 0x20 )
      v4 = 32;
    if ( v4 < (unsigned int)v5 )
    {
      sub_C8C990(a1 + 80, a2);
      goto LABEL_7;
    }
    memset(*(void **)(a1 + 88), -1, 8 * v5);
  }
  *(_QWORD *)(a1 + 100) = 0;
LABEL_7:
  *(_DWORD *)(a1 + 368) = 0;
  v6 = *(_QWORD *)(a1 + 8);
  if ( v6 )
  {
    v7 = *(void (**)(void))(*(_QWORD *)v6 + 32LL);
    if ( v7 != nullsub_321 )
      v7();
  }
  v8 = *(_QWORD *)(a1 + 16);
  if ( v8 )
  {
    v9 = *(void (**)(void))(*(_QWORD *)v8 + 16LL);
    if ( v9 != nullsub_324 )
      v9();
  }
  v10 = *(_QWORD *)(a1 + 24);
  if ( v10 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 16LL))(v10);
}
