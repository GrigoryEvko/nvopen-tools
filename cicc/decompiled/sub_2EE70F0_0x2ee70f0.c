// Function: sub_2EE70F0
// Address: 0x2ee70f0
//
unsigned __int64 __fastcall sub_2EE70F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int64 result; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdx

  v2 = a2 + 48;
  result = sub_2E313E0(a2);
  if ( result != a2 + 48 )
  {
    v6 = result;
    LODWORD(result) = *(_DWORD *)(a1 + 8);
    do
    {
      while ( 1 )
      {
        result = (unsigned int)result;
        v7 = (unsigned int)result + 1LL;
        if ( v7 > *(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v7, 8u, v4, v5);
          result = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = v6;
        result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
        *(_DWORD *)(a1 + 8) = result;
        if ( !v6 )
          BUG();
        if ( (*(_BYTE *)v6 & 4) == 0 )
          break;
        v6 = *(_QWORD *)(v6 + 8);
        if ( v2 == v6 )
          return result;
      }
      while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
        v6 = *(_QWORD *)(v6 + 8);
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v2 != v6 );
  }
  return result;
}
