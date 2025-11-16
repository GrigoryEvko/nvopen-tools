// Function: sub_2D22B40
// Address: 0x2d22b40
//
__int64 __fastcall sub_2D22B40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v8; // rax
  _QWORD *v9; // r14
  __int64 v10; // rcx
  int v11; // eax

  v3 = *(_QWORD *)(a2 + 16);
  v4 = *(_QWORD *)(a3 + 16);
  if ( v3 != v4 )
  {
    if ( (v5 = v3 & 0xFFFFFFFFFFFFFFF8LL, (v3 & 0xFFFFFFFFFFFFFFF8LL) == 0)
      || (v8 = v4 & 0xFFFFFFFFFFFFFFF8LL, (v9 = (_QWORD *)v8) == 0)
      || ((v10 = v3 >> 2, (v3 & 4) == 0)
       || *(_QWORD *)(v5 + 24) != *(_QWORD *)(v8 + 24)
       || *(_BYTE *)(v5 + 64) != *(_BYTE *)(v8 + 64)
       || (v11 = memcmp((const void *)(v5 + 40), (const void *)(v8 + 40), 0x18u), v10 = v3 >> 2, v11)
       || *(_QWORD *)(v5 + 72) != v9[9]
       || *(_QWORD *)(v5 + 80) != v9[10]
       || *(_QWORD *)(v5 + 88) != v9[11])
      && ((v10 & 1) != 0 || !sub_B46220(v3 & 0xFFFFFFFFFFFFFFF8LL, (__int64)v9)) )
    {
      v3 = 0;
    }
  }
  v6 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = v3;
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = v6;
  return a1;
}
