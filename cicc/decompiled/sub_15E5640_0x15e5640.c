// Function: sub_15E5640
// Address: 0x15e5640
//
__int64 __fastcall sub_15E5640(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        char a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v9; // rax
  int v10; // eax
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // rax

  v9 = sub_1646BA0(a2, a4);
  sub_1648CB0(a1, v9, a3);
  v10 = *(_DWORD *)(a1 + 20);
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 20) = v10 & 0xF0000000 | 1;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a1 + 32) & 0xFFFF8000LL | a5 & 0xF;
  if ( (a5 & 0xFu) - 7 <= 1 )
    *(_BYTE *)(a1 + 33) |= 0x40u;
  result = sub_164B780(a1, a6);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v12 = *(_QWORD *)(a1 - 16);
    result = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)result = v12;
    if ( v12 )
    {
      result |= *(_QWORD *)(v12 + 16) & 3LL;
      *(_QWORD *)(v12 + 16) = result;
    }
  }
  *(_QWORD *)(a1 - 24) = a7;
  if ( a7 )
  {
    v13 = *(_QWORD *)(a7 + 8);
    *(_QWORD *)(a1 - 16) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = (a1 - 16) | *(_QWORD *)(v13 + 16) & 3LL;
    result = (a7 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a1 - 24 + 16) = result;
    *(_QWORD *)(a7 + 8) = a1 - 24;
  }
  return result;
}
