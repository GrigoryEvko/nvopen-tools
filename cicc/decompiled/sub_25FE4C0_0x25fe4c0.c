// Function: sub_25FE4C0
// Address: 0x25fe4c0
//
unsigned __int64 *__fastcall sub_25FE4C0(unsigned __int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rcx
  __int64 v5; // rax
  unsigned __int64 *result; // rax

  v3 = a1[1];
  if ( v3 == a1[2] )
    return sub_25FE190(a1, v3, a2);
  if ( v3 )
  {
    *(_QWORD *)(v3 + 16) = 0;
    *(_QWORD *)(v3 + 8) = 0;
    *(_DWORD *)(v3 + 24) = 0;
    *(_QWORD *)v3 = 1;
    v4 = *(_QWORD *)(a2 + 8);
    ++*(_QWORD *)a2;
    v5 = *(_QWORD *)(v3 + 8);
    *(_QWORD *)(v3 + 8) = v4;
    LODWORD(v4) = *(_DWORD *)(a2 + 16);
    *(_QWORD *)(a2 + 8) = v5;
    LODWORD(v5) = *(_DWORD *)(v3 + 16);
    *(_DWORD *)(v3 + 16) = v4;
    LODWORD(v4) = *(_DWORD *)(a2 + 20);
    *(_DWORD *)(a2 + 16) = v5;
    LODWORD(v5) = *(_DWORD *)(v3 + 20);
    *(_DWORD *)(v3 + 20) = v4;
    LODWORD(v4) = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a2 + 20) = v5;
    result = (unsigned __int64 *)*(unsigned int *)(v3 + 24);
    *(_DWORD *)(v3 + 24) = v4;
    *(_DWORD *)(a2 + 24) = (_DWORD)result;
    v3 = a1[1];
  }
  a1[1] = v3 + 32;
  return result;
}
