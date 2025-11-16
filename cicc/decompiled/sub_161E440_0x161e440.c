// Function: sub_161E440
// Address: 0x161e440
//
__int64 __fastcall sub_161E440(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // ecx
  bool v4; // zf
  __int64 v5; // rdx
  char v6; // dl
  __int64 v7; // r8
  __int64 v8; // rdx
  int v9; // r8d
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 result; // rax

  v2 = *(unsigned int *)(a2 + 8);
  v3 = *(_DWORD *)(a2 + 8);
  *(_QWORD *)a1 = *(_QWORD *)(a2 + 8 * (1 - v2));
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8 * (2 - v2));
  v4 = *(_BYTE *)a2 == 15;
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 8 * (3 - v2));
  v5 = a2;
  if ( !v4 )
    v5 = *(_QWORD *)(a2 - 8 * v2);
  *(_QWORD *)(a1 + 24) = v5;
  *(_DWORD *)(a1 + 32) = *(_DWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 8 * (4 - v2));
  v6 = *(_BYTE *)(a2 + 40);
  *(_BYTE *)(a1 + 48) = (v6 & 4) != 0;
  *(_BYTE *)(a1 + 49) = (v6 & 8) != 0;
  *(_DWORD *)(a1 + 52) = *(_DWORD *)(a2 + 28);
  v7 = 0;
  if ( v3 > 8 )
    v7 = *(_QWORD *)(a2 + 8 * (8 - v2));
  *(_QWORD *)(a1 + 56) = v7;
  *(_BYTE *)(a1 + 80) = (v6 & 0x10) != 0;
  *(_DWORD *)(a1 + 64) = v6 & 3;
  v8 = *(_QWORD *)(a2 + 8 * (5 - v2));
  *(_QWORD *)(a1 + 68) = *(_QWORD *)(a2 + 32);
  v9 = *(_DWORD *)(a2 + 44);
  *(_QWORD *)(a1 + 88) = v8;
  *(_DWORD *)(a1 + 76) = v9;
  v10 = *(_QWORD *)(a2 + 8 * (6 - v2));
  v11 = *(_QWORD *)(a2 + 8 * (7 - v2));
  if ( v3 <= 9 )
  {
    *(_QWORD *)(a1 + 96) = 0;
    *(_QWORD *)(a1 + 104) = v10;
    *(_QWORD *)(a1 + 112) = v11;
    *(_QWORD *)(a1 + 120) = 0;
    return 0;
  }
  else
  {
    *(_QWORD *)(a1 + 104) = v10;
    *(_QWORD *)(a1 + 112) = v11;
    *(_QWORD *)(a1 + 96) = *(_QWORD *)(a2 + 8 * (9 - v2));
    if ( v3 == 10 )
      result = 0;
    else
      result = *(_QWORD *)(a2 + 8 * (10 - v2));
    *(_QWORD *)(a1 + 120) = result;
  }
  return result;
}
