// Function: sub_15E51E0
// Address: 0x15e51e0
//
__int64 __fastcall sub_15E51E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        char a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int16 a9,
        unsigned int a10,
        char a11)
{
  __int64 v12; // rax
  int v13; // eax
  char v14; // cl
  int v15; // edx
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 result; // rax
  __int64 v23; // rdx
  __int64 v24; // rax

  v12 = sub_1646BA0(a3, a10);
  sub_1648CB0(a1, v12, 3);
  v13 = *(_DWORD *)(a1 + 20);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 24) = a3;
  *(_DWORD *)(a1 + 20) = (a6 != 0) | v13 & 0xF0000000;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a1 + 32) & 0xFFFF8000LL | a5 & 0xF;
  if ( (a5 & 0xFu) - 7 <= 1 )
    *(_BYTE *)(a1 + 33) |= 0x40u;
  sub_164B780(a1, a7);
  v14 = *(_BYTE *)(a1 + 80);
  *(_QWORD *)(a1 + 48) = 0;
  v15 = *(_DWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = v14 & 0xFC | (a4 | (2 * a11)) & 3;
  *(_DWORD *)(a1 + 32) = v15 & 0x63FF | (a9 << 10) & 0x1C00;
  if ( a6 )
  {
    if ( *(_QWORD *)(a1 - 24) )
    {
      v16 = *(_QWORD *)(a1 - 16);
      v17 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v17 = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
    }
    v18 = *(_QWORD *)(a6 + 8);
    *(_QWORD *)(a1 - 24) = a6;
    *(_QWORD *)(a1 - 16) = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = (a1 - 16) | *(_QWORD *)(v18 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = *(_QWORD *)(a1 - 8) & 3LL | (a6 + 8);
    *(_QWORD *)(a6 + 8) = a1 - 24;
  }
  v19 = a1 + 56;
  if ( a8 )
  {
    sub_1631BE0(*(_QWORD *)(a8 + 40) + 8LL, a1);
    v20 = *(_QWORD *)(a8 + 56);
    v21 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(a1 + 64) = a8 + 56;
    v20 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 56) = v20 | v21 & 7;
    *(_QWORD *)(v20 + 8) = v19;
    result = v19 | *(_QWORD *)(a8 + 56) & 7LL;
    *(_QWORD *)(a8 + 56) = result;
  }
  else
  {
    sub_1631BE0(a2 + 8, a1);
    v23 = *(_QWORD *)(a2 + 8);
    v24 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(a1 + 64) = a2 + 8;
    v23 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 56) = v23 | v24 & 7;
    *(_QWORD *)(v23 + 8) = v19;
    result = v19 | *(_QWORD *)(a2 + 8) & 7LL;
    *(_QWORD *)(a2 + 8) = result;
  }
  return result;
}
