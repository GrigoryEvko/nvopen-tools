// Function: sub_E8DA50
// Address: 0xe8da50
//
__int64 __fastcall sub_E8DA50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  int v11; // eax
  __int64 result; // rax

  sub_E98820(a1, a2, a3);
  sub_E5CB20(*(_QWORD *)(a1 + 296), a2, v7, v8, v9, v10);
  v11 = *(unsigned __int8 *)(a2 + 9);
  *(_QWORD *)a2 = a4;
  *(_QWORD *)(a2 + 24) = a5;
  result = v11 & 0xFFFFFF8F | 0x10;
  *(_BYTE *)(a2 + 9) = result;
  return result;
}
