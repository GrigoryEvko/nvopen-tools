// Function: sub_389C450
// Address: 0x389c450
//
__int64 __fastcall sub_389C450(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 result; // rax
  bool v7; // zf
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rax
  _QWORD v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)v4 <= 0xD )
  {
    v5 = 10880;
    if ( _bittest64(&v5, v4) )
      return 0;
  }
  while ( 1 )
  {
    if ( a3 )
    {
      if ( !*(_BYTE *)(a3 + 4) && *(_DWORD *)(a1 + 64) == 87 )
      {
        *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
        v7 = *(_BYTE *)(a3 + 4) == 0;
        *(_DWORD *)a3 = *(_DWORD *)(a2 + 8);
        if ( v7 )
          *(_BYTE *)(a3 + 4) = 1;
      }
    }
    result = sub_389C3E0((__int64 **)a1, v11);
    if ( (_BYTE)result )
      break;
    v10 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v10 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v8, v9);
      v10 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v10) = v11[0];
    ++*(_DWORD *)(a2 + 8);
    if ( *(_DWORD *)(a1 + 64) != 4 )
      return 0;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  }
  return result;
}
