// Function: sub_B90650
// Address: 0xb90650
//
__int64 __fastcall sub_B90650(_QWORD *a1, __int64 a2)
{
  unsigned __int8 v2; // al
  _QWORD *v3; // rsi
  __int64 result; // rax

  v2 = *(_BYTE *)(a2 - 16);
  if ( (v2 & 2) != 0 )
  {
    *a1 = **(_QWORD **)(a2 - 32);
    a1[1] = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    a1[2] = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    v3 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    v3 = (_QWORD *)(a2 - 16 - 8LL * ((v2 >> 2) & 0xF));
    *a1 = *v3;
    a1[1] = v3[1];
    a1[2] = v3[2];
  }
  result = v3[3];
  a1[3] = result;
  return result;
}
