// Function: sub_38DCD40
// Address: 0x38dcd40
//
void __fastcall sub_38DCD40(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 i; // r13
  __int64 v5; // rsi
  __int64 (*v6)(); // rcx
  unsigned __int64 v7; // rdx

  v2 = *(_QWORD *)(a1 + 24);
  for ( i = *(_QWORD *)(a1 + 32);
        i != v2;
        *(_DWORD *)(v2 - 12) = ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64))v6)(a2, v5, v7) )
  {
    while ( 1 )
    {
      if ( a2 )
      {
        v5 = *(_QWORD *)(v2 + 32);
        v6 = *(__int64 (**)())(*(_QWORD *)a2 + 144LL);
        v7 = 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)(v2 + 40) - v5) >> 4);
        if ( v6 != sub_38DBAD0 )
          break;
      }
      *(_DWORD *)(v2 + 68) = 0;
      v2 += 80;
      if ( i == v2 )
        return;
    }
    v2 += 80;
  }
}
