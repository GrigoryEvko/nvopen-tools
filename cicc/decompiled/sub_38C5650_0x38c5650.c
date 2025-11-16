// Function: sub_38C5650
// Address: 0x38c5650
//
__int64 __fastcall sub_38C5650(__int64 a1, _QWORD *a2, _QWORD *a3, __int64 a4)
{
  unsigned int v5; // ebx
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  int v8; // r14d
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax

  v5 = a4;
  v6 = sub_16D3930(a3, a4);
  v7 = sub_1680880(a1 + 8, (__int64)a3, (v6 << 32) | v5);
  v8 = v7;
  if ( !*(_BYTE *)(a1 + 64) )
    return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 424LL))(a2, v7, 4);
  v10 = a2[1];
  v11 = sub_38CF310(*(_QWORD *)a1, 0, v10, 0);
  v12 = sub_38CB470(v8, v10);
  v13 = sub_38CB1F0(0, v11, v12, v10, 0);
  return sub_38DDD30(a2, v13, 4, 0);
}
