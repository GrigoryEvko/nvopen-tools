// Function: sub_1643FB0
// Address: 0x1643fb0
//
__int64 __fastcall sub_1643FB0(__int64 a1, const void *a2, __int64 a3, char a4)
{
  unsigned int v5; // eax
  unsigned int v6; // ecx
  __int64 result; // rax
  size_t v8; // r13
  __int64 v9; // rcx

  v5 = *(_DWORD *)(a1 + 8);
  v6 = v5 >> 8;
  if ( a4 )
  {
    result = ((v6 | 3) << 8) | (unsigned __int8)v5;
    *(_DWORD *)(a1 + 8) = result;
    *(_DWORD *)(a1 + 12) = a3;
    if ( !a3 )
      goto LABEL_3;
  }
  else
  {
    result = ((v6 | 1) << 8) | (unsigned __int8)v5;
    *(_DWORD *)(a1 + 8) = result;
    *(_DWORD *)(a1 + 12) = a3;
    if ( !a3 )
    {
LABEL_3:
      *(_QWORD *)(a1 + 16) = 0;
      return result;
    }
  }
  v8 = 8 * a3;
  result = sub_145CBF0((__int64 *)(**(_QWORD **)a1 + 2272LL), 8 * a3, 8);
  v9 = result;
  if ( v8 )
  {
    result = (__int64)memcpy((void *)result, a2, v8);
    v9 = result;
  }
  *(_QWORD *)(a1 + 16) = v9;
  return result;
}
