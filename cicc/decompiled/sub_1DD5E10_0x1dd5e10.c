// Function: sub_1DD5E10
// Address: 0x1dd5e10
//
__int64 __fastcall sub_1DD5E10(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 (*v4)(void); // rax
  __int64 v5; // r13
  unsigned __int64 v6; // rax
  __int64 (*v7)(); // rax

  v2 = 0;
  v3 = a2;
  v4 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(a1 + 56) + 16LL) + 40LL);
  if ( v4 != sub_1D00B00 )
    v2 = v4();
  if ( a1 + 24 != a2 )
  {
    v5 = 0x20000000303DLL;
    while ( 1 )
    {
      v6 = **(unsigned __int16 **)(v3 + 16);
      if ( (unsigned __int16)v6 > 0x2Du || !_bittest64(&v5, v6) )
      {
        v7 = *(__int64 (**)())(*(_QWORD *)v2 + 1016LL);
        if ( v7 == sub_1DD4CD0 || !((unsigned __int8 (__fastcall *)(__int64, __int64))v7)(v2, v3) )
          break;
      }
      if ( (*(_BYTE *)v3 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v3 + 46) & 8) != 0 )
          v3 = *(_QWORD *)(v3 + 8);
      }
      v3 = *(_QWORD *)(v3 + 8);
      if ( v3 == a1 + 24 )
        return v3;
    }
  }
  return v3;
}
