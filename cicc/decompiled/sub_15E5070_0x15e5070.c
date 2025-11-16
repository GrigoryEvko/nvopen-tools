// Function: sub_15E5070
// Address: 0x15e5070
//
__int64 __fastcall sub_15E5070(
        __int64 a1,
        __int64 a2,
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
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rax

  v11 = sub_1646BA0(a2, a8);
  sub_1648CB0(a1, v11, 3);
  v12 = *(_DWORD *)(a1 + 20);
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 20) = (a5 != 0) | v12 & 0xF0000000;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a1 + 32) & 0xFFFF8000LL | a4 & 0xF;
  if ( (a4 & 0xFu) - 7 <= 1 )
    *(_BYTE *)(a1 + 33) |= 0x40u;
  sub_164B780(a1, a6);
  v13 = *(_BYTE *)(a1 + 80);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  v14 = v13 & 0xFC | (a3 | (2 * a9)) & 3;
  v15 = *(_DWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = v14;
  result = v15 & 0x63FF | (a7 << 10) & 0x1C00u;
  *(_DWORD *)(a1 + 32) = result;
  if ( a5 )
  {
    if ( *(_QWORD *)(a1 - 24) )
    {
      v17 = *(_QWORD *)(a1 - 16);
      v18 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v18 = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
    }
    v19 = *(_QWORD *)(a5 + 8);
    *(_QWORD *)(a1 - 24) = a5;
    *(_QWORD *)(a1 - 16) = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = (a1 - 16) | *(_QWORD *)(v19 + 16) & 3LL;
    result = *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a1 - 24 + 16) = result | (a5 + 8);
    *(_QWORD *)(a5 + 8) = a1 - 24;
  }
  return result;
}
