// Function: sub_16A8750
// Address: 0x16a8750
//
void __fastcall sub_16A8750(_QWORD *a1, unsigned int a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  unsigned int v6; // r13d
  unsigned int v7; // ebx
  unsigned int v8; // r12d
  unsigned int v9; // esi
  unsigned int v11; // [rsp+8h] [rbp-38h]

  v6 = (a4 + 63) >> 6;
  v7 = a5 & 0x3F;
  v8 = (a4 + 63) & 0xFFFFFFC0;
  v11 = a5 >> 6;
  sub_16A7050((__int64)a1, a3 + 8LL * (a5 >> 6), v6);
  sub_16A8050(a1, v6, v7);
  v9 = v8 - v7;
  if ( a4 <= v8 - v7 )
  {
    if ( a4 < v9 && (a4 & 0x3F) != 0 )
      a1[v6 - 1] &= 0xFFFFFFFFFFFFFFFFLL >> (64 - (a4 & 0x3F));
  }
  else
  {
    a1[v6 - 1] |= (*(_QWORD *)(a3 + 8LL * (v6 + v11))
                 & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v8 - ((unsigned __int8)a4 + (unsigned __int8)v7) + 64))) << v9;
  }
  if ( v6 < a2 )
    memset(&a1[v6], 0, 8LL * (a2 - 1 - v6) + 8);
}
