// Function: sub_11A12D0
// Address: 0x11a12d0
//
_QWORD *__fastcall sub_11A12D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int16 v3; // ax

  v2 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = v2;
  if ( v2 != 0 && v2 != -4096 && v2 != -8192 )
    sub_BD73F0(a1 + 8);
  v3 = *(_WORD *)(a2 + 64);
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 56);
  *(_WORD *)(a1 + 40) = v3;
  return sub_B33910((_QWORD *)(a1 + 48), (__int64 *)a2);
}
