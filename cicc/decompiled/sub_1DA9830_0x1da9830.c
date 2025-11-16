// Function: sub_1DA9830
// Address: 0x1da9830
//
unsigned __int64 __fastcall sub_1DA9830(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  unsigned __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax

  v2 = *(_DWORD *)(a1 + 84);
  v3 = (unsigned __int64)sub_1DA89E0(*(_QWORD *)(a1 + 88));
  if ( v2 )
  {
    v4 = 0;
    do
    {
      *(_QWORD *)(v3 + 4 * v4) = *(_QWORD *)(a1 + 4 * v4);
      *(_QWORD *)(v3 + 4 * v4 + 8) = *(_QWORD *)(a1 + 4 * v4 + 8);
      *(_DWORD *)(v3 + v4 + 144) = *(_DWORD *)(a1 + v4 + 64);
      v4 += 4;
    }
    while ( v4 != 4LL * v2 );
  }
  v5 = v2 - 1;
  *(_DWORD *)(a1 + 80) = 1;
  v6 = v5 | v3 & 0xFFFFFFFFFFFFFFC0LL;
  *(_OWORD *)a1 = 0;
  *(_OWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_OWORD *)(a1 + 16) = 0;
  *(_OWORD *)(a1 + 48) = 0;
  v7 = *(_QWORD *)((v6 & 0xFFFFFFFFFFFFFFC0LL) + 16 * v5 + 8);
  *(_QWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 40) = v7;
  v8 = *(_QWORD *)(v6 & 0xFFFFFFFFFFFFFFC0LL);
  *(_DWORD *)(a1 + 84) = 1;
  *(_QWORD *)a1 = v8;
  return (unsigned __int64)a2 << 32;
}
