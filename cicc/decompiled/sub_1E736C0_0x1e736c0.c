// Function: sub_1E736C0
// Address: 0x1e736c0
//
unsigned __int16 *__fastcall sub_1E736C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int16 *result; // rax
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rsi
  unsigned __int16 *v10; // rcx
  int v11; // edi
  int v12; // esi
  int v13; // edx
  __int64 v14; // rax

  if ( *(_DWORD *)(a1 + 4) || (result = (unsigned __int16 *)*(unsigned int *)(a1 + 8), (_DWORD)result) )
  {
    v6 = *(_QWORD *)(a1 + 16);
    v7 = *(_QWORD *)(v6 + 24);
    if ( !v7 )
    {
      if ( (unsigned __int8)sub_1F4B670(a2 + 632) )
      {
        v14 = sub_1F4B8B0(a2 + 632, *(_QWORD *)(v6 + 8));
        *(_QWORD *)(v6 + 24) = v14;
        v7 = v14;
      }
      else
      {
        v7 = *(_QWORD *)(v6 + 24);
      }
    }
    v8 = *(unsigned __int16 *)(v7 + 2);
    v9 = *(_QWORD *)(*(_QWORD *)(a3 + 176) + 136LL);
    result = (unsigned __int16 *)(v9 + 4 * v8);
    v10 = (unsigned __int16 *)(v9 + 4 * (v8 + *(unsigned __int16 *)(v7 + 4)));
    if ( v10 != result )
    {
      v11 = *(_DWORD *)(a1 + 4);
      v12 = *(_DWORD *)(a1 + 8);
      do
      {
        v13 = *result;
        if ( v13 == v11 )
        {
          *(_DWORD *)(a1 + 40) += result[1];
          v13 = *result;
        }
        if ( v12 == v13 )
          *(_DWORD *)(a1 + 44) += result[1];
        result += 2;
      }
      while ( result != v10 );
    }
  }
  return result;
}
