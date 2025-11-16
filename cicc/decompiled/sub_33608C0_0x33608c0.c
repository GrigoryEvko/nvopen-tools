// Function: sub_33608C0
// Address: 0x33608c0
//
void __fastcall sub_33608C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 i; // r12
  int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rdx

  v6 = *(_QWORD *)(a1 + 592);
  v7 = *(_QWORD *)(v6 + 408);
  for ( i = v6 + 400; i != v7; v7 = *(_QWORD *)(v7 + 8) )
  {
    while ( 1 )
    {
      if ( v7 )
      {
        v9 = *(_DWORD *)(v7 + 16);
        if ( v9 < 0 )
        {
          v10 = *(_QWORD *)(a1 + 16);
          v11 = 40LL * (unsigned int)~v9;
          if ( (*(_BYTE *)(*(_QWORD *)(v10 + 8) - v11 + 26) & 8) != 0 )
            break;
        }
      }
      v7 = *(_QWORD *)(v7 + 8);
      if ( i == v7 )
        return;
    }
    sub_335F9F0(a1, v7 - 8, v11, v10, a5, a6);
  }
}
