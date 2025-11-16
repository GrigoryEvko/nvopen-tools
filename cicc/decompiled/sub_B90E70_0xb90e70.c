// Function: sub_B90E70
// Address: 0xb90e70
//
__int64 __fastcall sub_B90E70(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  int v3; // edx
  int v4; // ecx
  __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // rsi
  __int64 result; // rax

  v2 = *(_BYTE *)(a2 - 16);
  v3 = *(_DWORD *)(a2 + 20);
  v4 = *(_DWORD *)(a2 + 16);
  if ( (v2 & 2) != 0 )
  {
    *(_QWORD *)a1 = **(_QWORD **)(a2 - 32);
    v5 = *(_QWORD *)(a2 - 32);
    *(_DWORD *)(a1 + 16) = v4;
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(v5 + 8);
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    v6 = *(_QWORD *)(a2 - 32);
    *(_DWORD *)(a1 + 40) = v3;
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(v6 + 24);
    v7 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    *(_DWORD *)(a1 + 16) = v4;
    *(_DWORD *)(a1 + 40) = v3;
    v7 = (_QWORD *)(a2 - 16 - 8LL * ((v2 >> 2) & 0xF));
    *(_QWORD *)a1 = *v7;
    *(_QWORD *)(a1 + 8) = v7[1];
    *(_QWORD *)(a1 + 24) = v7[2];
    *(_QWORD *)(a1 + 32) = v7[3];
  }
  result = v7[4];
  *(_QWORD *)(a1 + 48) = result;
  return result;
}
