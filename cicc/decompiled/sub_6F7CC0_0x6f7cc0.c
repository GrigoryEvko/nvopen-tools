// Function: sub_6F7CC0
// Address: 0x6f7cc0
//
__int64 __fastcall sub_6F7CC0(
        const __m128i *a1,
        const __m128i *a2,
        const __m128i *a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 a7)
{
  char v9; // bl
  int v10; // eax
  int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v26; // [rsp+18h] [rbp-38h]

  v9 = a6;
  v26 = sub_6F6F40(a1, 0, (__int64)a3, a4, a7, a6);
  v10 = sub_8D2B80(a1->m128i_i64[0]);
  v11 = v9 & 1;
  v12 = sub_73DBF0(103 - ((unsigned int)(v10 == 0) - 1), a4, v26);
  v13 = (unsigned int)(16 * v11);
  v14 = v12;
  *(_BYTE *)(v12 + 59) = (16 * v11) | *(_BYTE *)(v12 + 59) & 0xEF;
  v15 = *(_QWORD *)(v12 + 72);
  *(_QWORD *)(v15 + 16) = sub_6F6F40(a2, 0, v16, v17, v18, v13);
  v19 = *(_QWORD *)(*(_QWORD *)(v14 + 72) + 16LL);
  *(_QWORD *)(v19 + 16) = sub_6F6F40(a3, 0, v20, v21, v22, v23);
  if ( a5 )
  {
    *(_BYTE *)(v14 + 25) |= 1u;
    *(_BYTE *)(v14 + 58) |= 1u;
  }
  return sub_6E7170((__int64 *)v14, a7);
}
