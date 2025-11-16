// Function: sub_1ECC910
// Address: 0x1ecc910
//
_QWORD *__fastcall sub_1ECC910(__int64 a1, const void **a2)
{
  int v3; // eax
  _QWORD *v5; // rdi
  int v6; // esi
  _QWORD *result; // rax
  size_t v8; // rdx

  v3 = *((_DWORD *)a2 + 1);
  v5 = (_QWORD *)(a1 + 8);
  v6 = *(_DWORD *)a2;
  *((_DWORD *)v5 - 1) = v3;
  *((_DWORD *)v5 - 2) = v6;
  result = sub_1ECC890(v5, (unsigned int)(v3 * v6));
  v8 = 4LL * (unsigned int)(*(_DWORD *)(a1 + 4) * *(_DWORD *)a1);
  if ( v8 )
    return memmove(*(void **)(a1 + 8), a2[1], v8);
  return result;
}
