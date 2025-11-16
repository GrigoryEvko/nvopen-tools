// Function: sub_25499E0
// Address: 0x25499e0
//
__int64 __fastcall sub_25499E0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 result; // rax
  int v6; // esi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r12

  result = *(unsigned __int8 *)(a1 + 97);
  if ( (*(_BYTE *)(a1 + 97) & 3) == 3 )
  {
    v6 = 50;
  }
  else
  {
    v6 = 51;
    if ( (result & 2) == 0 )
    {
      if ( (result & 1) == 0 )
        return result;
      v6 = 78;
    }
  }
  v9 = sub_A778C0(a3, v6, 0);
  result = *(unsigned int *)(a4 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), result + 1, 8u, v7, v8);
    result = *(unsigned int *)(a4 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = v9;
  ++*(_DWORD *)(a4 + 8);
  return result;
}
