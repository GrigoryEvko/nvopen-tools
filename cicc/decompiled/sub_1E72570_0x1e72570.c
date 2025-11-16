// Function: sub_1E72570
// Address: 0x1e72570
//
unsigned __int64 __fastcall sub_1E72570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  _DWORD *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 result; // rax
  __int64 v11; // rdx
  __int64 i; // rdx

  v7 = *(_DWORD **)(a1 + 152);
  if ( v7 && v7[2] )
  {
    (*(void (__fastcall **)(_DWORD *))(*(_QWORD *)v7 + 8LL))(v7);
    *(_QWORD *)(a1 + 152) = 0;
  }
  v8 = *(_QWORD *)(a1 + 64);
  if ( v8 != *(_QWORD *)(a1 + 72) )
    *(_QWORD *)(a1 + 72) = v8;
  v9 = *(_QWORD *)(a1 + 128);
  if ( v9 != *(_QWORD *)(a1 + 136) )
    *(_QWORD *)(a1 + 136) = v9;
  *(_BYTE *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 172) = 0xFFFFFFFFLL;
  result = *(unsigned int *)(a1 + 200);
  *(_QWORD *)(a1 + 164) = 0;
  *(_QWORD *)(a1 + 180) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_BYTE *)(a1 + 280) = 0;
  *(_DWORD *)(a1 + 296) = 0;
  if ( result <= 1 )
  {
    if ( result )
      return result;
    if ( !*(_DWORD *)(a1 + 204) )
    {
      sub_16CD150(a1 + 192, (const void *)(a1 + 208), 1u, 4, a5, a6);
      result = *(unsigned int *)(a1 + 200);
    }
    v11 = *(_QWORD *)(a1 + 192);
    result = v11 + 4 * result;
    for ( i = v11 + 4; i != result; result += 4LL )
    {
      if ( result )
        *(_DWORD *)result = 0;
    }
  }
  *(_DWORD *)(a1 + 200) = 1;
  return result;
}
