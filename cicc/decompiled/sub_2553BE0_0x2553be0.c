// Function: sub_2553BE0
// Address: 0x2553be0
//
__int64 __fastcall sub_2553BE0(__int64 a1, __m128i *a2, __int64 a3, char a4)
{
  unsigned __int64 v5; // rbx
  char v6; // al
  unsigned __int8 v7; // dl
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rax
  __int64 v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = sub_250D070(a2);
  v6 = sub_2509800(a2);
  v7 = *(_BYTE *)v5;
  if ( v6 == 7 )
  {
    a4 = 1;
  }
  else if ( v7 == 60 )
  {
    return 1;
  }
  if ( (unsigned int)v7 - 12 <= 1 )
    return 1;
  if ( v7 == 20 )
  {
    v9 = *(_QWORD *)(v5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
      v9 = **(_QWORD **)(v9 + 16);
    v10 = *(_DWORD *)(v9 + 8);
    v11 = sub_25096F0(a2);
    if ( !sub_B2F070(v11, v10 >> 8) )
      return 1;
  }
  v12[0] = 0x1600000051LL;
  return sub_2516400(a1, a2, (__int64)v12, 2, a4, 22);
}
