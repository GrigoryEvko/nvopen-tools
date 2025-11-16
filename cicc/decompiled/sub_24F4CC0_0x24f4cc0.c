// Function: sub_24F4CC0
// Address: 0x24f4cc0
//
__int64 __fastcall sub_24F4CC0(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4)
{
  int v4; // eax
  unsigned __int64 v5; // r12
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // rsi
  int v11; // edx
  __int64 result; // rax
  unsigned __int64 v13; // [rsp+18h] [rbp-68h] BYREF
  int v14; // [rsp+28h] [rbp-58h]
  _BYTE v15[32]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v16; // [rsp+50h] [rbp-30h]

  v4 = *(_DWORD *)(a1 + 280);
  v13 = a3;
  if ( (unsigned int)(v4 - 1) > 1 )
    BUG();
  v5 = *(_QWORD *)(a1 + 344);
  v16 = 257;
  v8 = sub_24F2F10(
         a2,
         0x31u,
         v13,
         *(__int64 ***)(*(_QWORD *)(*(_QWORD *)(v5 + 24) + 16LL) + 8LL),
         (__int64)v15,
         0,
         v14,
         0);
  v16 = 257;
  v9 = *(_QWORD *)(v5 + 24);
  v13 = v8;
  v10 = sub_921880((unsigned int **)a2, v9, v5, (int)&v13, 1, (__int64)v15, 0);
  v11 = *(_WORD *)(v10 + 2) & 0xF003;
  result = v11 | (4 * ((*(_WORD *)(v5 + 2) >> 4) & 0x3FFu));
  *(_WORD *)(v10 + 2) = v11 | (4 * ((*(_WORD *)(v5 + 2) >> 4) & 0x3FF));
  if ( a4 )
    return sub_24F49B0(a4, v10, v5);
  return result;
}
