// Function: sub_B90730
// Address: 0xb90730
//
__int64 __fastcall sub_B90730(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  _QWORD *v3; // rdx
  __int64 result; // rax
  __int64 v5; // rax

  *(_DWORD *)a1 = (unsigned __int16)sub_AF18C0(a2);
  v2 = *(_BYTE *)(a2 - 16);
  if ( (v2 & 2) != 0 )
  {
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 24LL);
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 32LL);
    v3 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    v5 = 8LL * ((v2 >> 2) & 0xF);
    v3 = (_QWORD *)(a2 - 16 - v5);
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 - v5);
    *(_QWORD *)(a1 + 16) = v3[3];
    *(_QWORD *)(a1 + 24) = v3[4];
  }
  *(_QWORD *)(a1 + 32) = v3[5];
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 24);
  *(_DWORD *)(a1 + 48) = sub_AF18D0(a2);
  result = *(unsigned int *)(a2 + 44);
  *(_DWORD *)(a1 + 52) = result;
  return result;
}
