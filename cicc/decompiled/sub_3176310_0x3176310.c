// Function: sub_3176310
// Address: 0x3176310
//
__int64 (__fastcall *__fastcall sub_3176310(__int64 a1, __int64 a2))(__int64, __int64, __int64)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  void (__fastcall *v5)(__int64, __int64, __int64); // rax
  void (__fastcall *v6)(__int64, __int64, __int64); // rax
  void (__fastcall *v7)(__int64, __int64, __int64); // rax
  __int64 (__fastcall *result)(__int64, __int64, __int64); // rax

  sub_3176190(a1, a2);
  sub_3176100(a1);
  sub_C7D6A0(*(_QWORD *)(a1 + 768), 16LL * *(unsigned int *)(a1 + 784), 8);
  v2 = *(unsigned int *)(a1 + 752);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 736);
    v4 = v3 + 144 * v2;
    do
    {
      if ( *(_QWORD *)v3 != -8192 && *(_QWORD *)v3 != -4096 )
        sub_C7D6A0(*(_QWORD *)(v3 + 72), 24LL * *(unsigned int *)(v3 + 88), 8);
      v3 += 144;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 752);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 736), 144 * v2, 8);
  if ( *(_BYTE *)(a1 + 468) )
  {
    if ( *(_BYTE *)(a1 + 180) )
      goto LABEL_10;
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 448));
    if ( *(_BYTE *)(a1 + 180) )
      goto LABEL_10;
  }
  _libc_free(*(_QWORD *)(a1 + 160));
LABEL_10:
  v5 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 136);
  if ( v5 )
    v5(a1 + 120, a1 + 120, 3);
  v6 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 104);
  if ( v6 )
    v6(a1 + 88, a1 + 88, 3);
  v7 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 72);
  if ( v7 )
    v7(a1 + 56, a1 + 56, 3);
  result = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a1 + 40);
  if ( result )
    return (__int64 (__fastcall *)(__int64, __int64, __int64))result(a1 + 24, a1 + 24, 3);
  return result;
}
