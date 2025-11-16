// Function: sub_B90CD0
// Address: 0xb90cd0
//
__int64 __fastcall sub_B90CD0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  int v3; // edx
  char v4; // r8
  char v5; // cl
  int v6; // r9d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rsi
  __int64 result; // rax

  v2 = *(_BYTE *)(a2 - 16);
  v3 = *(_DWORD *)(a2 + 4);
  v4 = *(_BYTE *)(a2 + 20);
  v5 = *(_BYTE *)(a2 + 21);
  v6 = *(_DWORD *)(a2 + 16);
  if ( (v2 & 2) != 0 )
  {
    *(_QWORD *)a1 = **(_QWORD **)(a2 - 32);
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 40LL);
    v7 = *(_QWORD *)(a2 - 32);
    *(_DWORD *)(a1 + 32) = v6;
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(v7 + 16);
    v8 = *(_QWORD *)(a2 - 32);
    *(_BYTE *)(a1 + 48) = v4;
    v9 = *(_QWORD *)(v8 + 24);
    *(_BYTE *)(a1 + 49) = v5;
    *(_QWORD *)(a1 + 40) = v9;
    *(_QWORD *)(a1 + 56) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 48LL);
    v10 = *(_QWORD *)(a2 - 32);
    *(_DWORD *)(a1 + 72) = v3;
    *(_QWORD *)(a1 + 64) = *(_QWORD *)(v10 + 56);
    v11 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    *(_DWORD *)(a1 + 32) = v6;
    *(_BYTE *)(a1 + 48) = v4;
    *(_BYTE *)(a1 + 49) = v5;
    v11 = (_QWORD *)(a2 - 16 - 8LL * ((v2 >> 2) & 0xF));
    *(_DWORD *)(a1 + 72) = v3;
    *(_QWORD *)a1 = *v11;
    *(_QWORD *)(a1 + 8) = v11[1];
    *(_QWORD *)(a1 + 16) = v11[5];
    *(_QWORD *)(a1 + 24) = v11[2];
    *(_QWORD *)(a1 + 40) = v11[3];
    *(_QWORD *)(a1 + 56) = v11[6];
    *(_QWORD *)(a1 + 64) = v11[7];
  }
  result = v11[8];
  *(_QWORD *)(a1 + 80) = result;
  return result;
}
