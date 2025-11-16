// Function: sub_E10A10
// Address: 0xe10a10
//
__int64 __fastcall sub_E10A10(__int64 a1, __int64 a2)
{
  _BYTE *v4; // r13
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  void *v7; // rdi
  __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // r13
  __int64 result; // rax

  v4 = *(_BYTE **)(a1 + 24);
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v4 + 32LL))(v4);
  if ( (v4[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 40LL))(v4, a2);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_QWORD *)(a2 + 16);
  v7 = *(void **)a2;
  v8 = v5 + 1;
  if ( v5 + 1 > v6 )
  {
    v9 = v5 + 993;
    v10 = 2 * v6;
    if ( v9 > v10 )
      *(_QWORD *)(a2 + 16) = v9;
    else
      *(_QWORD *)(a2 + 16) = v10;
    v11 = realloc(v7);
    *(_QWORD *)a2 = v11;
    v7 = (void *)v11;
    if ( !v11 )
      abort();
    v5 = *(_QWORD *)(a2 + 8);
    v8 = v5 + 1;
  }
  *(_QWORD *)(a2 + 8) = v8;
  *((_BYTE *)v7 + v5) = 64;
  v12 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v12 + 32LL))(v12, a2);
  result = v12[9] & 0xC0;
  if ( (v12[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v12 + 40LL))(v12, a2);
  return result;
}
