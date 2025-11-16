// Function: sub_B90C30
// Address: 0xb90c30
//
__int64 __fastcall sub_B90C30(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  _QWORD *v3; // rdx
  __int64 result; // rax

  *(_DWORD *)a1 = (unsigned __int16)sub_AF18C0(a2);
  v2 = *(_BYTE *)(a2 - 16);
  if ( (v2 & 2) != 0 )
  {
    *(_QWORD *)(a1 + 8) = **(_QWORD **)(a2 - 32);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    *(_BYTE *)(a1 + 24) = *(_BYTE *)(a2 + 1) >> 7;
    v3 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    v3 = (_QWORD *)(a2 - 16 - 8LL * ((v2 >> 2) & 0xF));
    *(_QWORD *)(a1 + 8) = *v3;
    *(_QWORD *)(a1 + 16) = v3[1];
    *(_BYTE *)(a1 + 24) = *(_BYTE *)(a2 + 1) >> 7;
  }
  result = v3[2];
  *(_QWORD *)(a1 + 32) = result;
  return result;
}
