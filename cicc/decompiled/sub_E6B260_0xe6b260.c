// Function: sub_E6B260
// Address: 0xe6b260
//
unsigned __int64 __fastcall sub_E6B260(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r12

  v2 = a1[36];
  a1[46] += 208LL;
  v3 = (v2 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[37] >= v3 + 208 && v2 )
    a1[36] = v3 + 208;
  else
    v3 = sub_9D1E70((__int64)(a1 + 36), 208, 208, 3);
  sub_E81B30(v3, 1, 0);
  *(_BYTE *)(v3 + 30) = 0;
  *(_QWORD *)(v3 + 40) = v3 + 64;
  *(_QWORD *)(v3 + 96) = v3 + 112;
  *(_QWORD *)(v3 + 32) = 0;
  *(_QWORD *)(v3 + 48) = 0;
  *(_QWORD *)(v3 + 56) = 32;
  *(_QWORD *)(v3 + 104) = 0x400000000LL;
  *(_QWORD *)(v3 + 8) = a2;
  **(_QWORD **)(a2 + 8) = v3;
  *(_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL) = v3;
  return v3;
}
