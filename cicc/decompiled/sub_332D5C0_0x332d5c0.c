// Function: sub_332D5C0
// Address: 0x332d5c0
//
void __fastcall sub_332D5C0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        unsigned int a7,
        __int64 a8)
{
  int v9; // edx
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // r8
  __int64 v13; // r9

  v9 = *(_DWORD *)(a2 + 24);
  if ( v9 > 239 )
  {
    v10 = (unsigned int)(v9 - 242) < 2 ? 0x28 : 0;
  }
  else
  {
    v10 = 40;
    if ( v9 <= 237 )
      v10 = (unsigned int)(v9 - 101) < 0x30 ? 0x28 : 0;
  }
  v11 = sub_2FE5730(
          *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + v10) + 48LL)
                   + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + v10 + 8)),
          0,
          a3,
          a4,
          a5,
          a6,
          a7);
  sub_332D1E0(a1, a2, v11, a8, v12, v13);
}
