// Function: sub_2850770
// Address: 0x2850770
//
char __fastcall sub_2850770(
        __int64 *a1,
        __int64 a2,
        char a3,
        __int64 a4,
        char a5,
        unsigned int a6,
        __int64 a7,
        unsigned int a8,
        __int64 a9)
{
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // r15
  char result; // al
  __int64 v13; // [rsp+10h] [rbp-48h]
  char v15; // [rsp+1Eh] [rbp-3Ah]

  v10 = *(_QWORD *)(a9 + 32);
  v11 = *(_QWORD *)a9;
  v15 = *(_BYTE *)(a9 + 16);
  v13 = *(_QWORD *)(a9 + 8);
  result = sub_2850670(a1, a2, a3, a4, a5, a6, a7, a8, *(_QWORD *)a9, v13, v15, *(_BYTE *)(a9 + 24), v10);
  if ( !result && v10 == 1 )
    return sub_2850670(a1, a2, a3, a4, a5, a6, a7, a8, v11, v13, v15, 1u, 0);
  return result;
}
