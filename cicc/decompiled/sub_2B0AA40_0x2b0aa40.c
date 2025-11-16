// Function: sub_2B0AA40
// Address: 0x2b0aa40
//
__int64 __fastcall sub_2B0AA40(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r11
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v9; // r15
  __int64 v10; // r13

  v4 = a1;
  v6 = ((char *)a2 - (char *)a1) >> 4;
  v7 = 2 * a4;
  if ( 2 * a4 <= v6 )
  {
    v9 = 4 * a4;
    v10 = -2 * a4;
    do
    {
      a3 = sub_2B09EB0(v4, &v4[v9 + v10], &v4[v9 + v10], &v4[v9], a3);
      v6 = ((char *)a2 - (char *)v4) >> 4;
    }
    while ( v7 <= v6 );
  }
  if ( a4 <= v6 )
    v6 = a4;
  return sub_2B09EB0(v4, &v4[2 * v6], &v4[2 * v6], a2, a3);
}
