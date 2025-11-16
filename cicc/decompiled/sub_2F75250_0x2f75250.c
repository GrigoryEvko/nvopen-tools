// Function: sub_2F75250
// Address: 0x2f75250
//
void __fastcall sub_2F75250(__int64 a1, _DWORD *a2)
{
  int v3; // r13d
  unsigned int v4; // r12d
  __int64 v5; // rax
  unsigned __int64 v6; // rdi

  v3 = *(_DWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)a2 + 16LL))
                 + 16);
  v4 = v3 + a2[16];
  if ( v4 < *(_DWORD *)(a1 + 216) >> 2 || v4 > *(_DWORD *)(a1 + 216) )
  {
    v5 = (__int64)_libc_calloc(v4, 1u);
    if ( !v5 && (v4 || (v5 = malloc(1u)) == 0) )
      sub_C64F00("Allocation failed", 1u);
    v6 = *(_QWORD *)(a1 + 208);
    *(_QWORD *)(a1 + 208) = v5;
    if ( v6 )
      _libc_free(v6);
    *(_DWORD *)(a1 + 216) = v4;
  }
  *(_DWORD *)(a1 + 224) = v3;
}
