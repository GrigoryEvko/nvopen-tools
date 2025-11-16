// Function: sub_320EDF0
// Address: 0x320edf0
//
void __fastcall sub_320EDF0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rbx
  __int64 v9; // rsi
  __int64 v10; // [rsp+8h] [rbp-38h]

  v5 = *(__int64 **)a2;
  v10 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( v10 != *(_QWORD *)a2 )
  {
    do
    {
      v9 = *v5++;
      sub_320E0C0(a1, v9, a3, a4, a5);
    }
    while ( (__int64 *)v10 != v5 );
  }
}
