// Function: sub_1995490
// Address: 0x1995490
//
char __fastcall sub_1995490(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7)
{
  unsigned int v11; // ecx
  __int64 v12; // rbx
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // r8
  char result; // al
  char v16; // al
  unsigned int v17; // [rsp+14h] [rbp-4Ch]
  unsigned __int64 v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h]

  v11 = a6;
  v12 = *(_QWORD *)(a7 + 8);
  v13 = *(_QWORD *)(a7 + 24);
  v14 = *(_QWORD *)a7;
  if ( v12 < a2 + v12 == a2 > 0 && v12 < a3 + v12 == a3 > 0 )
  {
    v17 = *(unsigned __int8 *)(a7 + 16);
    v18 = *(_QWORD *)a7;
    v19 = a5;
    v16 = sub_1992C60(a1, a4, a5, a6, v14, a2 + v12, v17, *(_QWORD *)(a7 + 24));
    a5 = v19;
    v11 = a6;
    v14 = v18;
    if ( v16 )
    {
      result = sub_1992C60(a1, a4, v19, a6, v18, a3 + v12, v17, v13);
      if ( result )
        return result;
      a5 = v19;
      v11 = a6;
      v14 = v18;
    }
  }
  if ( v13 == 1 )
    return sub_1993620(a1, a2, a3, a4, a5, v11, v14, v12, 1u, 0);
  else
    return 0;
}
