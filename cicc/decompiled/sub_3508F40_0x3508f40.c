// Function: sub_3508F40
// Address: 0x3508f40
//
unsigned __int64 __fastcall sub_3508F40(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // r13d
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rbx
  unsigned __int64 result; // rax
  unsigned __int64 *v14; // rax
  unsigned __int64 *v15; // rdx

  v3 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a2 + 32) + 32LL) + 16LL);
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 200LL))(v3);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)a1 = v4;
  v8 = *(_DWORD *)(v4 + 16);
  v9 = *(_DWORD *)(a1 + 56) >> 2;
  if ( v8 < (unsigned int)v9 || v8 > *(_DWORD *)(a1 + 56) )
  {
    v10 = (__int64)_libc_calloc(v8, 1u);
    if ( !v10 && (v8 || (v10 = malloc(1u)) == 0) )
      sub_C64F00("Allocation failed", 1u);
    v11 = *(_QWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 48) = v10;
    if ( v11 )
      _libc_free(v11);
    *(_DWORD *)(a1 + 56) = v8;
  }
  sub_35085F0((_QWORD *)a1, a2, v9, v5, v6, v7);
  v12 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v12 )
LABEL_24:
    BUG();
  result = *(_QWORD *)v12;
  if ( (*(_QWORD *)v12 & 4) == 0 && (*(_BYTE *)(v12 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      result &= 0xFFFFFFFFFFFFFFF8LL;
      v12 = result;
      if ( (*(_BYTE *)(result + 44) & 4) == 0 )
        break;
      result = *(_QWORD *)result;
    }
  }
  while ( a2 + 48 != v12 )
  {
    sub_3508F10((_QWORD *)a1, v12);
    v14 = (unsigned __int64 *)(*(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL);
    v15 = v14;
    if ( !v14 )
      goto LABEL_24;
    v12 = *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL;
    result = *v14;
    if ( (result & 4) == 0 && (*((_BYTE *)v15 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        result &= 0xFFFFFFFFFFFFFFF8LL;
        v12 = result;
        if ( (*(_BYTE *)(result + 44) & 4) == 0 )
          break;
        result = *(_QWORD *)result;
      }
    }
  }
  return result;
}
