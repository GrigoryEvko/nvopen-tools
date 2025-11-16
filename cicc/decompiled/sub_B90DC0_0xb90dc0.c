// Function: sub_B90DC0
// Address: 0xb90dc0
//
__int64 __fastcall sub_B90DC0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  int v3; // r8d
  int v4; // ecx
  int v5; // edx
  int v6; // r9d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rsi
  __int64 result; // rax

  v2 = *(_BYTE *)(a2 - 16);
  v3 = *(unsigned __int16 *)(a2 + 20);
  v4 = *(_DWORD *)(a2 + 24);
  v5 = *(_DWORD *)(a2 + 4);
  v6 = *(_DWORD *)(a2 + 16);
  if ( (v2 & 2) != 0 )
  {
    *(_QWORD *)a1 = **(_QWORD **)(a2 - 32);
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    v7 = *(_QWORD *)(a2 - 32);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(v7 + 16);
    v8 = *(_QWORD *)(a2 - 32);
    *(_DWORD *)(a1 + 40) = v3;
    v9 = *(_QWORD *)(v8 + 24);
    *(_DWORD *)(a1 + 44) = v4;
    *(_DWORD *)(a1 + 48) = v5;
    *(_QWORD *)(a1 + 32) = v9;
    v10 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    *(_DWORD *)(a1 + 24) = v6;
    *(_DWORD *)(a1 + 40) = v3;
    *(_DWORD *)(a1 + 44) = v4;
    v10 = (_QWORD *)(a2 - 16 - 8LL * ((v2 >> 2) & 0xF));
    *(_DWORD *)(a1 + 48) = v5;
    *(_QWORD *)a1 = *v10;
    *(_QWORD *)(a1 + 8) = v10[1];
    *(_QWORD *)(a1 + 16) = v10[2];
    *(_QWORD *)(a1 + 32) = v10[3];
  }
  result = v10[4];
  *(_QWORD *)(a1 + 56) = result;
  return result;
}
