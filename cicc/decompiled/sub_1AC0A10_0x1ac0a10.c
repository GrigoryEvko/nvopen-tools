// Function: sub_1AC0A10
// Address: 0x1ac0a10
//
__int64 __fastcall sub_1AC0A10(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  char v8; // al
  __int64 result; // rax

  v7 = a1 + 40;
  v8 = byte_4FB6320;
  *(_QWORD *)(v7 - 24) = a5;
  *(_QWORD *)(v7 - 16) = a6;
  if ( !a4 )
    a4 = v8;
  *(_QWORD *)(v7 - 40) = a2;
  *(_BYTE *)(v7 - 8) = 0;
  *(_BYTE *)(v7 - 32) = a4;
  result = sub_1ABFDF0(
             v7,
             *(__int64 **)(a3 + 32),
             (__int64)(*(_QWORD *)(a3 + 40) - *(_QWORD *)(a3 + 32)) >> 3,
             a2,
             0,
             0);
  *(_DWORD *)(a1 + 96) = -1;
  return result;
}
