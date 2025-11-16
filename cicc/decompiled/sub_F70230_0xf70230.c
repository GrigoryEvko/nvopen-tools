// Function: sub_F70230
// Address: 0xf70230
//
unsigned __int8 *__fastcall sub_F70230(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdi

  v5 = (unsigned int)(a1 - 1);
  if ( (unsigned int)v5 > 0xF )
    BUG();
  return sub_F700B0(dword_3F8AE00[v5], a2, a3, a4, a5);
}
