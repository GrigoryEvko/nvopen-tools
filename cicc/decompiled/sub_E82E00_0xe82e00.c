// Function: sub_E82E00
// Address: 0xe82e00
//
__int64 __fastcall sub_E82E00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // ebx
  int v8; // eax
  __int64 result; // rax

  v7 = (((unsigned __int8)a4 + 1) << 15) & 0xF8000 | 0x3020;
  sub_E5CB20(*(_QWORD *)(a1 + 296), a2, a3, a4, a5, a6);
  v8 = *(_DWORD *)(a2 + 8);
  *(_QWORD *)(a2 + 24) = a3;
  result = v8 & 0xFFF00FDF;
  *(_DWORD *)(a2 + 8) = result | v7;
  return result;
}
