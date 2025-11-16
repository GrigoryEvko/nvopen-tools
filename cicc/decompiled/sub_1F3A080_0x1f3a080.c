// Function: sub_1F3A080
// Address: 0x1f3a080
//
__int64 __fastcall sub_1F3A080(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 result; // rax

  v10 = (__int64)sub_1E0B7C0(*(_QWORD *)(a2 + 56), a6);
  sub_1E17170(v10, *(_DWORD *)(*(_QWORD *)(v10 + 32) + 8LL), a4, a5, a7);
  sub_1DD5BA0((__int64 *)(a2 + 16), v10);
  v11 = *a3;
  v12 = *(_QWORD *)v10;
  *(_QWORD *)(v10 + 8) = a3;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v10 = v11 | v12 & 7;
  *(_QWORD *)(v11 + 8) = v10;
  result = *a3 & 7;
  *a3 = result | v10;
  return result;
}
