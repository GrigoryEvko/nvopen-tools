// Function: sub_E15D50
// Address: 0xe15d50
//
unsigned __int64 __fastcall sub_E15D50(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  void *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  _BYTE *v10; // r13
  __int64 v11; // rsi
  unsigned __int64 result; // rax
  void *v13; // rdi
  __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax

  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(void **)a2;
  if ( v4 + 1 > v5 )
  {
    v7 = v4 + 993;
    v8 = 2 * v5;
    if ( v7 > v8 )
      *(_QWORD *)(a2 + 16) = v7;
    else
      *(_QWORD *)(a2 + 16) = v8;
    v9 = realloc(v6);
    *(_QWORD *)a2 = v9;
    v6 = (void *)v9;
    if ( !v9 )
      goto LABEL_25;
    v4 = *(_QWORD *)(a2 + 8);
  }
  *((_BYTE *)v6 + v4) = 32;
  ++*(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(a1 + 24) || *(_QWORD *)(a1 + 32) )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 123);
  }
  v10 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v10 + 32LL))(v10, a2);
  if ( (v10[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v10 + 40LL))(v10, a2);
  if ( *(_BYTE *)(a1 + 24) || *(_QWORD *)(a1 + 32) )
  {
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 125);
    if ( *(_BYTE *)(a1 + 24) )
      sub_E12F20((__int64 *)a2, 9u, " noexcept");
    if ( *(_QWORD *)(a1 + 32) )
    {
      sub_E12F20((__int64 *)a2, 4u, " -> ");
      sub_E15BE0(*(_BYTE **)(a1 + 32), a2);
    }
  }
  v11 = *(_QWORD *)(a2 + 8);
  result = *(_QWORD *)(a2 + 16);
  v13 = *(void **)a2;
  v14 = v11 + 1;
  if ( v11 + 1 > result )
  {
    v15 = v11 + 993;
    v16 = 2 * result;
    if ( v15 > v16 )
      *(_QWORD *)(a2 + 16) = v15;
    else
      *(_QWORD *)(a2 + 16) = v16;
    result = realloc(v13);
    *(_QWORD *)a2 = result;
    v13 = (void *)result;
    if ( result )
    {
      v11 = *(_QWORD *)(a2 + 8);
      v14 = v11 + 1;
      goto LABEL_22;
    }
LABEL_25:
    abort();
  }
LABEL_22:
  *(_QWORD *)(a2 + 8) = v14;
  *((_BYTE *)v13 + v11) = 59;
  return result;
}
