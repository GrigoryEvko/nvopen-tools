// Function: sub_1B90E50
// Address: 0x1b90e50
//
__int64 __fastcall sub_1B90E50(__int64 a1, __int64 a2, _QWORD *a3)
{
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __int64 *v7; // rdi
  __int64 result; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v12; // [rsp+10h] [rbp-20h] BYREF

  sub_16E2FC0(v11, a2);
  v5 = (_BYTE *)v11[0];
  *(_BYTE *)(a1 + 8) = 0;
  v6 = v11[1];
  *(_QWORD *)a1 = &unk_49F6D50;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  sub_1B8E960((__int64 *)(a1 + 16), v5, (__int64)&v5[v6]);
  v7 = (__int64 *)v11[0];
  *(_QWORD *)(a1 + 56) = a1 + 72;
  *(_QWORD *)(a1 + 64) = 0x100000000LL;
  *(_QWORD *)(a1 + 88) = 0x100000000LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 104) = 0;
  if ( v7 != &v12 )
    j_j___libc_free_0(v7, v12 + 1);
  *(_QWORD *)a1 = &unk_49F7110;
  result = a1 + 112;
  *(_QWORD *)(a1 + 120) = a1 + 112;
  *(_QWORD *)(a1 + 112) = (a1 + 112) | 4;
  if ( a3 )
  {
    a3[2] = result;
    v9 = a3[1];
    v10 = (a1 + 112) & 0xFFFFFFFFFFFFFFF8LL;
    a3[4] = a1;
    *(_QWORD *)(v10 + 8) = a3 + 1;
    a3[1] = v10 | v9 & 7;
    result = *(_QWORD *)(a1 + 112) & 7LL;
    *(_QWORD *)(a1 + 112) = result | (unsigned __int64)(a3 + 1);
  }
  return result;
}
