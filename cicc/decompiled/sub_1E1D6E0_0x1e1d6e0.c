// Function: sub_1E1D6E0
// Address: 0x1e1d6e0
//
void __fastcall sub_1E1D6E0(__int64 a1, unsigned __int64 *a2, __int64 a3, unsigned __int64 *a4)
{
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // r12
  __int64 *v7; // rdi
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rax

  if ( a2 != a4 )
  {
    if ( !a4 )
      BUG();
    v5 = a4;
    if ( (*(_BYTE *)a4 & 4) == 0 && (*((_BYTE *)a4 + 46) & 8) != 0 )
    {
      do
        v5 = (unsigned __int64 *)v5[1];
      while ( (*((_BYTE *)v5 + 46) & 8) != 0 );
    }
    v6 = (unsigned __int64 *)v5[1];
    if ( a4 != v6 && a2 != v6 )
    {
      v7 = (__int64 *)(a1 + 16);
      if ( v7 != (__int64 *)(a3 + 16) )
        sub_1DD5C00(v7, a3 + 16, (__int64)a4, v5[1]);
      if ( v6 != a2 && v6 != a4 )
      {
        v8 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*a4 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v6;
        *v6 = *v6 & 7 | *a4 & 0xFFFFFFFFFFFFFFF8LL;
        v9 = *a2;
        *(_QWORD *)(v8 + 8) = a2;
        v9 &= 0xFFFFFFFFFFFFFFF8LL;
        *a4 = v9 | *a4 & 7;
        *(_QWORD *)(v9 + 8) = a4;
        *a2 = v8 | *a2 & 7;
      }
    }
  }
}
