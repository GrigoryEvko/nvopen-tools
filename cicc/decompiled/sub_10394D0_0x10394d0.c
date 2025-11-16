// Function: sub_10394D0
// Address: 0x10394d0
//
void __fastcall sub_10394D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // r12

  sub_1038AF0(a3, *(char **)(a3 + 8), *(char **)(a2 + 8), *(char **)(a2 + 16));
  for ( i = *(_QWORD *)(a2 + 56); a2 + 40 != i; i = sub_220EEE0(i) )
    sub_10394D0(a1, *(_QWORD *)(i + 40), a3);
}
