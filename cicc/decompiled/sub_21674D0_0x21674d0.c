// Function: sub_21674D0
// Address: 0x21674d0
//
__int64 __fastcall sub_21674D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int8 v3; // al
  unsigned int v4; // edx
  unsigned int v5; // r8d
  char v7; // al

  v2 = *(_QWORD *)(a1 + 24);
  v3 = sub_2167220(*(_QWORD *)(a1 + 8), a2);
  v4 = 1;
  if ( (v3 == 1 || (v5 = 4, v3) && (v4 = v3, *(_QWORD *)(v2 + 8LL * v3 + 120)))
    && (v5 = 1, v7 = *(_BYTE *)(v2 + 259LL * v4 + 2498), (v7 & 0xFB) != 0) )
  {
    return 3 * (unsigned int)(v7 != 1) + 1;
  }
  else
  {
    return v5;
  }
}
