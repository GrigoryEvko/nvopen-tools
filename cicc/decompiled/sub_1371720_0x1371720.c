// Function: sub_1371720
// Address: 0x1371720
//
__int64 __fastcall sub_1371720(unsigned __int64 a1, __int16 a2, unsigned __int64 a3, __int16 a4)
{
  int v6; // r15d
  int v7; // eax
  int v9; // [rsp+Ch] [rbp-34h]

  if ( !a1 )
    return (unsigned int)-(a3 != 0);
  if ( !a3 )
    return 1;
  v6 = a4;
  v9 = sub_1371700(a1, a2);
  v7 = sub_1371700(a3, v6);
  if ( v9 != v7 )
    return 2 * (unsigned int)(v9 >= v7) - 1;
  if ( a2 >= a4 )
    return (unsigned int)-sub_16CB620(a3, a1, (unsigned int)(a2 - v6));
  return sub_16CB620(a1, a3, (unsigned int)(v6 - a2));
}
