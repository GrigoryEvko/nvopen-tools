// Function: sub_D1D5E0
// Address: 0xd1d5e0
//
__int64 (__fastcall *__fastcall sub_D1D5E0(__int64 a1))(__int64, __int64, __int64)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  unsigned __int64 v7; // r14
  __int64 v8; // rsi
  __int64 (__fastcall *result)(__int64, __int64, __int64); // rax

  v1 = *(_QWORD **)(a1 + 336);
  while ( (_QWORD *)(a1 + 336) != v1 )
  {
    v2 = v1;
    v1 = (_QWORD *)*v1;
    v3 = v2[5];
    v2[2] = &unk_49DB368;
    if ( v3 != 0 && v3 != -4096 && v3 != -8192 )
      sub_BD60C0(v2 + 3);
    j_j___libc_free_0(v2, 64);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 312), 16LL * *(unsigned int *)(a1 + 328), 8);
  v4 = *(unsigned int *)(a1 + 296);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD **)(a1 + 280);
    v6 = &v5[2 * v4];
    do
    {
      while ( 1 )
      {
        if ( *v5 != -4096 && *v5 != -8192 )
        {
          v7 = v5[1] & 0xFFFFFFFFFFFFFFF8LL;
          if ( v7 )
            break;
        }
        v5 += 2;
        if ( v6 == v5 )
          goto LABEL_15;
      }
      if ( (*(_BYTE *)(v7 + 8) & 1) != 0 )
      {
        j_j___libc_free_0(v5[1] & 0xFFFFFFFFFFFFFFF8LL, 272);
      }
      else
      {
        sub_C7D6A0(*(_QWORD *)(v7 + 16), 16LL * *(unsigned int *)(v7 + 24), 8);
        j_j___libc_free_0(v7, 272);
      }
      v5 += 2;
    }
    while ( v6 != v5 );
LABEL_15:
    v4 = *(unsigned int *)(a1 + 296);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 280), 16 * v4, 8);
  v8 = 16LL * *(unsigned int *)(a1 + 264);
  sub_C7D6A0(*(_QWORD *)(a1 + 248), v8, 8);
  if ( !*(_BYTE *)(a1 + 172) )
    _libc_free(*(_QWORD *)(a1 + 152), v8);
  if ( *(_BYTE *)(a1 + 68) )
  {
    result = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a1 + 24);
    if ( !result )
      return result;
    return (__int64 (__fastcall *)(__int64, __int64, __int64))result(a1 + 8, a1 + 8, 3);
  }
  _libc_free(*(_QWORD *)(a1 + 48), v8);
  result = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a1 + 24);
  if ( result )
    return (__int64 (__fastcall *)(__int64, __int64, __int64))result(a1 + 8, a1 + 8, 3);
  return result;
}
