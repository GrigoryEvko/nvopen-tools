// Function: sub_3258EA0
// Address: 0x3258ea0
//
__int64 __fastcall sub_3258EA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _DWORD *v3; // rax
  __int64 v4; // rdx
  __int64 result; // rax

  sub_3252AB0((_QWORD *)a1, a2);
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = off_4A36020;
  *(_WORD *)(a1 + 28) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  v2 = sub_31DA930(a2);
  v3 = sub_AE2980(v2, 0);
  v4 = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(a1 + 27) = v3[1] == 64;
  *(_BYTE *)(a1 + 28) = (unsigned int)(*(_DWORD *)(*(_QWORD *)(v4 + 200) + 544LL) - 3) <= 2;
  result = (unsigned int)(*(_DWORD *)(*(_QWORD *)(v4 + 200) + 544LL) - 36);
  *(_BYTE *)(a1 + 29) = (unsigned int)result <= 1;
  return result;
}
