// Function: sub_B2C3B0
// Address: 0xb2c3b0
//
__int64 __fastcall sub_B2C3B0(__int64 a1, __int64 a2, char a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rsi
  __int64 v11; // rax
  int v12; // eax
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rcx
  int v18; // eax
  unsigned __int64 v19; // rdx
  unsigned int v20; // r12d
  char v21; // dl
  __int64 result; // rax
  __int64 v23; // rax
  int v24; // r15d
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // r15

  v9 = a4;
  if ( a4 == -1 )
  {
    v9 = 0;
    if ( a6 )
      v9 = *(unsigned int *)(a6 + 320);
  }
  v11 = sub_BCE3C0(*(_QWORD *)a2, v9);
  sub_BD35F0(a1, v11, 0);
  v12 = *(_DWORD *)(a1 + 4);
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 4) = v12 & 0x38000000 | 0x40000000;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a1 + 32) & 0xFFFE0000LL | a3 & 0xF;
  if ( (a3 & 0xFu) - 7 <= 1 )
    *(_BYTE *)(a1 + 33) |= 0x40u;
  sub_BD6B50(a1, a5);
  *(_WORD *)(a1 + 34) &= 1u;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = (a1 + 72) | 4;
  *(_QWORD *)(a1 + 80) = a1 + 72;
  v13 = *(_DWORD *)(a2 + 12);
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = (unsigned int)(v13 - 1);
  *(_QWORD *)(a1 + 112) = 0;
  LOBYTE(v13) = qword_4F80F48[8];
  *(_QWORD *)(a1 + 120) = 0;
  *(_DWORD *)(a1 + 132) = -1;
  *(_BYTE *)(a1 + 128) = v13;
  v14 = sub_B2BE50(a1);
  if ( !(unsigned __int8)sub_B6F8E0(v14) )
  {
    v24 = qword_4F815C8;
    v25 = sub_22077B0(32);
    v26 = v25;
    if ( v25 )
    {
      sub_C926D0(v25, 0, 16);
      *(_DWORD *)(v26 + 24) = v24;
      *(_DWORD *)(v26 + 28) = 0;
    }
    v27 = *(_QWORD *)(a1 + 112);
    *(_QWORD *)(a1 + 112) = v26;
    if ( v27 )
    {
      sub_BD84F0(v27);
      j_j___libc_free_0(v27, 32);
    }
  }
  if ( *(_DWORD *)(a2 + 12) != 1 )
    *(_WORD *)(a1 + 2) = 1;
  if ( a6 )
  {
    sub_BA8540(a6 + 24, a1);
    v15 = *(_QWORD *)(a6 + 24);
    v16 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(a1 + 64) = a6 + 24;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 56) = v15 | v16 & 7;
    *(_QWORD *)(v15 + 8) = a1 + 56;
    LOBYTE(v16) = *(_BYTE *)(a6 + 872);
    *(_QWORD *)(a6 + 24) = *(_QWORD *)(a6 + 24) & 7LL | (a1 + 56);
    *(_BYTE *)(a1 + 128) = v16;
  }
  v17 = sub_BD5D20(a1);
  v18 = 0;
  if ( v19 > 4 )
  {
    if ( *(_DWORD *)v17 != 1836477548 || (v18 = 0, *(_BYTE *)(v17 + 4) != 46) )
      v18 = 1;
    LOBYTE(v18) = v18 == 0;
  }
  v20 = *(_DWORD *)(a1 + 36);
  v21 = 32 * v18;
  result = (32 * v18) | *(_BYTE *)(a1 + 33) & 0xDFu;
  *(_BYTE *)(a1 + 33) = v21 | *(_BYTE *)(a1 + 33) & 0xDF;
  if ( v20 )
  {
    v23 = sub_B2BE50(a1);
    result = sub_B612D0(v23, v20);
    *(_QWORD *)(a1 + 120) = result;
  }
  return result;
}
