// Function: sub_1DC2B40
// Address: 0x1dc2b40
//
unsigned __int64 __fastcall sub_1DC2B40(__int64 a1, _QWORD *a2)
{
  __int64 (*v2)(); // rax
  __int64 v3; // rax
  __int64 v4; // rcx
  int v5; // r8d
  int v6; // r9d
  unsigned int v7; // r14d
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // r8d
  int v13; // r9d
  unsigned __int64 v14; // rbx
  unsigned __int64 result; // rax
  unsigned __int64 *v16; // rax

  v2 = *(__int64 (**)())(**(_QWORD **)(**(_QWORD **)(a2[7] + 40LL) + 16LL) + 112LL);
  if ( v2 == sub_1D00B10 )
  {
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 16) = 0;
    BUG();
  }
  v3 = v2();
  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)a1 = v3;
  v7 = *(_DWORD *)(v3 + 16);
  v8 = *(_DWORD *)(a1 + 64) >> 2;
  if ( v7 < (unsigned int)v8 || v7 > *(_DWORD *)(a1 + 64) )
  {
    _libc_free(*(_QWORD *)(a1 + 56));
    v9 = (__int64)_libc_calloc(v7, 1u);
    if ( !v9 )
    {
      if ( v7 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v9 = 0;
      }
      else
      {
        v9 = sub_13A3880(1u);
      }
    }
    *(_QWORD *)(a1 + 56) = v9;
    *(_DWORD *)(a1 + 64) = v7;
  }
  sub_1DC29C0((__int64 *)a1, a2, v8, v4, v5, v6);
  v14 = a2[3] & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v14 )
    BUG();
  result = *(_QWORD *)v14;
  if ( (*(_QWORD *)v14 & 4) == 0 && (*(_BYTE *)(v14 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      result &= 0xFFFFFFFFFFFFFFF8LL;
      v14 = result;
      if ( (*(_BYTE *)(result + 46) & 4) == 0 )
        break;
      result = *(_QWORD *)result;
    }
  }
  while ( a2 + 3 != (_QWORD *)v14 )
  {
    sub_1DC2260((__int64 *)a1, v14, v10, v11, v12, v13);
    v16 = (unsigned __int64 *)(*(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL);
    v10 = (__int64)v16;
    if ( !v16 )
      BUG();
    v14 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
    result = *v16;
    if ( (result & 4) == 0 && (*(_BYTE *)(v10 + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        result &= 0xFFFFFFFFFFFFFFF8LL;
        v14 = result;
        if ( (*(_BYTE *)(result + 46) & 4) == 0 )
          break;
        result = *(_QWORD *)result;
      }
    }
  }
  return result;
}
