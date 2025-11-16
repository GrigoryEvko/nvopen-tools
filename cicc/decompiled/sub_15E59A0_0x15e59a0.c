// Function: sub_15E59A0
// Address: 0x15e59a0
//
__int64 __fastcall sub_15E59A0(__int64 a1, __int64 a2, unsigned int a3, char a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // [rsp-8h] [rbp-30h]

  sub_15E5640(a1, a2, 2u, a3, a4, a5, a6);
  result = v10;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  if ( a7 )
  {
    ((void (__fastcall *)(__int64, __int64))sub_1631CE0)(a7 + 56, a1);
    v8 = *(_QWORD *)(a7 + 56);
    v9 = *(_QWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 56) = a7 + 56;
    v8 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 48) = v8 | v9 & 7;
    *(_QWORD *)(v8 + 8) = a1 + 48;
    result = *(_QWORD *)(a7 + 56) & 7LL;
    *(_QWORD *)(a7 + 56) = result | (a1 + 48);
  }
  return result;
}
