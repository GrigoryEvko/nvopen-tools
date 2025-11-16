// Function: sub_3502D00
// Address: 0x3502d00
//
unsigned int *__fastcall sub_3502D00(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  int v8; // r12d
  int v9; // edx
  unsigned int *v10; // r14
  __int64 v12; // r13
  __int64 v13; // r12

  v6 = *(_BYTE *)(*(_QWORD *)(a1 + 24) + a2);
  if ( v6 <= 0x1Fu && (v7 = 720LL * v6, *(_DWORD *)(a1 + v7 + 48) == a2) )
  {
    v12 = *(_QWORD *)a1;
    v13 = *(_QWORD *)(a1 + 8);
    v10 = (unsigned int *)(a1 + v7 + 48);
    if ( !sub_3501B90(v10, v13, *(_QWORD *)a1) )
      sub_3501B00(v10, v13, v12);
  }
  else
  {
    v8 = *(_DWORD *)(a1 + 40);
    if ( v8 == 31 )
    {
      *(_DWORD *)(a1 + 40) = 0;
      LOBYTE(v8) = 31;
    }
    else
    {
      *(_DWORD *)(a1 + 40) = v8 + 1;
    }
    v9 = 32;
    while ( *(_DWORD *)(a1 + 720LL * (unsigned __int8)v8 + 56) )
    {
      LOBYTE(v8) = v8 + 1;
      if ( (_BYTE)v8 == 32 )
        LOBYTE(v8) = 0;
      if ( !--v9 )
        BUG();
    }
    v10 = (unsigned int *)(a1 + 720LL * (unsigned __int8)v8 + 48);
    sub_35028B0(v10, a2, *(_QWORD *)(a1 + 8), *(_QWORD *)a1, *(_QWORD *)(a1 + 16), a6);
    *(_BYTE *)(*(_QWORD *)(a1 + 24) + a2) = v8;
  }
  return v10;
}
