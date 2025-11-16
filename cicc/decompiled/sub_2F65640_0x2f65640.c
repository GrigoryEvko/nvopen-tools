// Function: sub_2F65640
// Address: 0x2f65640
//
void __fastcall sub_2F65640(__int64 a1, unsigned __int64 *a2, __int64 a3, unsigned __int64 *a4)
{
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax

  if ( a2 != a4 )
  {
    if ( !a4 )
      BUG();
    v5 = a4;
    if ( (*(_BYTE *)a4 & 4) == 0 && (*((_BYTE *)a4 + 44) & 8) != 0 )
    {
      do
        v5 = (unsigned __int64 *)v5[1];
      while ( (*((_BYTE *)v5 + 44) & 8) != 0 );
    }
    v6 = (unsigned __int64 *)v5[1];
    if ( a4 != v6 && a2 != v6 )
    {
      sub_2E310C0((__int64 *)(a1 + 40), (__int64 *)(a3 + 40), (__int64)a4, v5[1]);
      if ( v6 != a4 )
      {
        v7 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*a4 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v6;
        *v6 = *v6 & 7 | *a4 & 0xFFFFFFFFFFFFFFF8LL;
        v8 = *a2;
        *(_QWORD *)(v7 + 8) = a2;
        v8 &= 0xFFFFFFFFFFFFFFF8LL;
        *a4 = v8 | *a4 & 7;
        *(_QWORD *)(v8 + 8) = a4;
        *a2 = v7 | *a2 & 7;
      }
    }
  }
}
