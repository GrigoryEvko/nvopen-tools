// Function: sub_336F670
// Address: 0x336f670
//
__int64 __fastcall sub_336F670(__int64 a1, __int64 a2, __int16 a3, __int16 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  _QWORD *v8; // rdi
  unsigned int v9; // r13d
  size_t v11; // rdx
  __int64 v12; // rax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000001LL;
  v7 = a1 + 104;
  v8 = (_QWORD *)(a1 + 128);
  *(v8 - 6) = v7;
  *((_WORD *)v8 - 56) = a4;
  *(v8 - 13) = a5;
  *(v8 - 4) = 4;
  *((_WORD *)v8 - 12) = a3;
  *(v8 - 5) = 1;
  *(_QWORD *)(a1 + 112) = v8;
  *(_QWORD *)(a1 + 120) = 0x400000000LL;
  v9 = *(_DWORD *)(a2 + 8);
  if ( v9 && a1 + 112 != a2 )
  {
    v11 = 4LL * v9;
    if ( v9 <= 4
      || (sub_C8D5F0(a1 + 112, v8, v9, 4u, a1 + 112, v9),
          v8 = *(_QWORD **)(a1 + 112),
          v12 = *(unsigned int *)(a2 + 8),
          (v11 = 4 * v12) != 0) )
    {
      memcpy(v8, *(const void **)a2, v11);
      LODWORD(v12) = *(_DWORD *)(a2 + 8);
    }
    *(_DWORD *)(a1 + 120) = v9;
    v9 = v12;
  }
  *(_DWORD *)(a1 + 160) = v9;
  *(_QWORD *)(a1 + 144) = a1 + 160;
  *(_QWORD *)(a1 + 152) = 0x400000001LL;
  *(_QWORD *)(a1 + 176) = a6;
  return a6;
}
