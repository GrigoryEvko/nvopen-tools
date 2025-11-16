// Function: sub_B2FEA0
// Address: 0xb2fea0
//
__int64 __fastcall sub_B2FEA0(
        __int64 a1,
        _QWORD *a2,
        char a3,
        char a4,
        __int64 a5,
        __int64 a6,
        __int16 a7,
        unsigned int a8,
        char a9)
{
  __int64 v11; // rax
  int v12; // eax
  char v13; // dl
  char v14; // r12
  int v15; // edx
  __int64 result; // rax
  __int64 v17; // rax

  v11 = sub_BCE3C0(*a2, a8);
  sub_BD35F0(a1, v11, 3);
  v12 = *(_DWORD *)(a1 + 4);
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 4) = v12 & 0x38000000 | 1;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a1 + 32) & 0xFFFE0000LL | a4 & 0xF;
  if ( (a4 & 0xFu) - 7 <= 1 )
    *(_BYTE *)(a1 + 33) |= 0x40u;
  sub_BD6B50(a1, a6);
  v13 = *(_BYTE *)(a1 + 80);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  v14 = v13 & 0xFC | (a3 | (2 * a9)) & 3;
  v15 = *(_DWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = v14;
  result = v15 & 0x1E3FF | (a7 << 10) & 0x1C00u;
  *(_DWORD *)(a1 + 32) = result;
  if ( a5 )
  {
    if ( *(_QWORD *)(a1 - 32) )
    {
      v17 = *(_QWORD *)(a1 - 24);
      **(_QWORD **)(a1 - 16) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = *(_QWORD *)(a1 - 16);
    }
    result = *(_QWORD *)(a5 + 16);
    *(_QWORD *)(a1 - 32) = a5;
    *(_QWORD *)(a1 - 24) = result;
    if ( result )
      *(_QWORD *)(result + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a5 + 16;
    *(_QWORD *)(a5 + 16) = a1 - 32;
  }
  else
  {
    *(_DWORD *)(a1 + 4) &= 0xF8000000;
  }
  return result;
}
