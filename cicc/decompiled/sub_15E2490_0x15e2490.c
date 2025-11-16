// Function: sub_15E2490
// Address: 0x15e2490
//
__int64 __fastcall sub_15E2490(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rcx
  int v16; // eax
  unsigned __int64 v17; // rdx
  int v18; // r12d
  char v19; // dl
  __int64 result; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // r15
  __int64 *v25; // rax

  v9 = sub_1646BA0(a2, 0);
  sub_1648CB0(a1, v9, 0);
  v10 = *(_QWORD *)(a1 + 32);
  *(_DWORD *)(a1 + 20) &= 0xF0000000;
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 32) = (unsigned int)v10 & 0xFFFF8000 | (unsigned __int64)(a3 & 0xF);
  if ( (a3 & 0xFu) - 7 <= 1 )
    *(_BYTE *)(a1 + 33) |= 0x40u;
  sub_164B780(a1, a4);
  *(_DWORD *)(a1 + 32) &= 0x7FFFu;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = (a1 + 72) | 4;
  *(_QWORD *)(a1 + 80) = a1 + 72;
  v11 = *(_DWORD *)(a2 + 12);
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 96) = (unsigned int)(v11 - 1);
  *(_QWORD *)(a1 + 112) = 0;
  v12 = sub_15E0530(a1);
  if ( !(unsigned __int8)sub_16033A0(v12) )
  {
    v22 = sub_22077B0(40);
    v23 = v22;
    if ( v22 )
    {
      sub_16D1950(v22, 0, 16);
      *(_DWORD *)(v23 + 32) = 0;
    }
    v24 = *(_QWORD *)(a1 + 104);
    *(_QWORD *)(a1 + 104) = v23;
    if ( v24 )
    {
      sub_164D180(v24);
      j_j___libc_free_0(v24, 40);
    }
  }
  if ( *(_DWORD *)(a2 + 12) != 1 )
    *(_WORD *)(a1 + 18) = 1;
  if ( a5 )
  {
    *(_QWORD *)(a1 + 40) = a5;
    if ( (*(_BYTE *)(a1 + 23) & 0x20) != 0 )
    {
      v21 = *(_QWORD *)(a5 + 120);
      if ( v21 )
        sub_164D6D0(v21, a1);
    }
    v13 = *(_QWORD *)(a5 + 24);
    v14 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(a1 + 64) = a5 + 24;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 56) = v13 | v14 & 7;
    *(_QWORD *)(v13 + 8) = a1 + 56;
    *(_QWORD *)(a5 + 24) = *(_QWORD *)(a5 + 24) & 7LL | (a1 + 56);
  }
  v15 = sub_1649960(a1);
  v16 = 0;
  if ( v17 > 4 )
  {
    if ( *(_DWORD *)v15 != 1836477548 || (v16 = 0, *(_BYTE *)(v15 + 4) != 46) )
      v16 = 1;
    LOBYTE(v16) = v16 == 0;
  }
  v18 = *(_DWORD *)(a1 + 36);
  v19 = 32 * v16;
  result = (32 * v16) | *(_BYTE *)(a1 + 33) & 0xDFu;
  *(_BYTE *)(a1 + 33) = v19 | *(_BYTE *)(a1 + 33) & 0xDF;
  if ( v18 )
  {
    v25 = (__int64 *)sub_15E0530(a1);
    result = sub_15E1850(v25, v18);
    *(_QWORD *)(a1 + 112) = result;
  }
  return result;
}
