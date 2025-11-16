// Function: sub_DF6350
// Address: 0xdf6350
//
__int64 __fastcall sub_DF6350(__int64 a1, unsigned int a2)
{
  __int64 v3; // [rsp-8h] [rbp-8h]

  if ( a2 > 1 )
    BUG();
  *((_BYTE *)&v3 - 4) = 0;
  return *(&v3 - 1);
}
