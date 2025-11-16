// Function: sub_2965130
// Address: 0x2965130
//
unsigned __int64 __fastcall sub_2965130(_QWORD *a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // r8

  v1 = a1[7];
  a1[17] += 160LL;
  v2 = (v1 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[8] >= v2 + 160 && v1 )
    a1[7] = v2 + 160;
  else
    v2 = sub_9D1E70((__int64)(a1 + 7), 160, 160, 3);
  memset((void *)v2, 0, 0xA0u);
  *(_QWORD *)(v2 + 72) = 8;
  *(_QWORD *)(v2 + 64) = v2 + 88;
  *(_BYTE *)(v2 + 84) = 1;
  return v2;
}
