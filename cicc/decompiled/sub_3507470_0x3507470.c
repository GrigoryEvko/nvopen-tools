// Function: sub_3507470
// Address: 0x3507470
//
void __fastcall sub_3507470(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 *i; // r12
  __int64 v9; // rbx
  __int64 v10; // r14

  v6 = a2;
  v7 = *(_QWORD *)(a2 + 104);
  for ( i = (__int64 *)a1[4]; v7; v7 = *(_QWORD *)(v7 + 104) )
  {
    v9 = *(_QWORD *)(v7 + 64);
    v10 = v9 + 8LL * *(unsigned int *)(v7 + 72);
    while ( v10 != v9 )
    {
      while ( 1 )
      {
        a2 = *(_QWORD *)(*(_QWORD *)v9 + 8LL);
        if ( (a2 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (a2 & 6) != 0 )
          break;
        v9 += 8;
        if ( v10 == v9 )
          goto LABEL_8;
      }
      v9 += 8;
      sub_2E0E0B0(v6, a2, i, a4, a5, a6);
    }
LABEL_8:
    ;
  }
  sub_2E1D8A0((__int64)a1, a2, a3, a4, a5, a6);
  sub_3507070(a1, (__int64 *)v6, *(_DWORD *)(v6 + 112), -1, -1, v6);
}
