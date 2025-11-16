// Function: sub_B90AE0
// Address: 0xb90ae0
//
__int64 __fastcall sub_B90AE0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  _QWORD *v3; // rdx
  __int64 result; // rax
  __int64 v5; // rax

  v2 = *(_BYTE *)(a2 - 16);
  if ( (v2 & 2) != 0 )
  {
    *(_QWORD *)a1 = **(_QWORD **)(a2 - 32);
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    v3 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    v5 = 8LL * ((v2 >> 2) & 0xF);
    v3 = (_QWORD *)(a2 - 16 - v5);
    *(_QWORD *)a1 = *v3;
    *(_QWORD *)(a1 + 8) = v3[1];
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 - v5);
  }
  *(_QWORD *)(a1 + 24) = v3[3];
  result = *(unsigned int *)(a2 + 4);
  *(_DWORD *)(a1 + 32) = result;
  return result;
}
