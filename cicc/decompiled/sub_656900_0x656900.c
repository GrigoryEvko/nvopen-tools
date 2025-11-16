// Function: sub_656900
// Address: 0x656900
//
__int64 __fastcall sub_656900(__int64 a1)
{
  unsigned int v1; // r14d
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rax
  const char *v6; // rsi
  int v7; // [rsp+18h] [rbp-2D8h] BYREF
  unsigned int v8; // [rsp+1Ch] [rbp-2D4h] BYREF
  __int64 v9; // [rsp+20h] [rbp-2D0h] BYREF
  __int64 v10; // [rsp+28h] [rbp-2C8h] BYREF
  _BYTE v11[8]; // [rsp+30h] [rbp-2C0h] BYREF
  __int64 v12; // [rsp+38h] [rbp-2B8h]
  _BYTE v13[8]; // [rsp+70h] [rbp-280h] BYREF
  _BYTE v14[16]; // [rsp+78h] [rbp-278h] BYREF
  __int64 v15; // [rsp+88h] [rbp-268h]
  char v16; // [rsp+B0h] [rbp-240h]
  char v17; // [rsp+B1h] [rbp-23Fh]
  __int64 v18; // [rsp+C0h] [rbp-230h]
  _QWORD v19[66]; // [rsp+E0h] [rbp-210h] BYREF

  v1 = dword_4F04C3C;
  sub_87E690(a1, 11);
  if ( (dword_4F077C4 == 1
     || dword_4F077C0
     && qword_4F077A8 <= 0x76BFu
     && *(_BYTE *)(a1 + 80) == 11
     && (v6 = *(const char **)(*(_QWORD *)a1 + 8LL)) != 0
     && !strcmp(v6, "exit"))
    && *(_DWORD *)(a1 + 40) != unk_4F066A8 )
  {
    sub_881DB0(a1);
    sub_885FF0(a1, 0, 1);
  }
  sub_7296C0(&v8);
  dword_4F04C3C = 1;
  v2 = sub_7259C0(7);
  **(_QWORD **)(v2 + 168) = 0;
  if ( dword_4F077C4 == 2 )
  {
    *(_QWORD *)(v2 + 160) = sub_72C930(7);
    *(_BYTE *)(*(_QWORD *)(v2 + 168) + 16LL) |= 2u;
    *(_BYTE *)(*(_QWORD *)(v2 + 168) + 16LL) |= 1u;
  }
  else
  {
    *(_QWORD *)(v2 + 160) = sub_72BA30(5);
    *(_BYTE *)(*(_QWORD *)(v2 + 168) + 16LL) &= ~2u;
  }
  sub_878710(a1, v11);
  sub_87E3B0(v13);
  v16 |= 0x40u;
  v18 = sub_624310(v2, (__int64)v13);
  if ( dword_4D048B8 )
    v15 = v12;
  memset(v19, 0, 0x1D8u);
  v19[19] = v19;
  v19[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v19[22]) |= 1u;
  WORD2(v19[33]) = 257;
  v19[36] = v2;
  v19[0] = a1;
  sub_6523A0((__int64)v11, (__int64)v19, (__int64)v13, 129, &v7, &v9, &v10, 0);
  if ( dword_4F04C64 == -1
    || (v3 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v3 + 7) & 1) == 0)
    || dword_4F04C44 == -1 && (*(_BYTE *)(v3 + 6) & 2) == 0 )
  {
    if ( (v17 & 8) == 0 )
      sub_87E280(v14);
  }
  v4 = v19[0];
  *(_BYTE *)(*(_QWORD *)(v19[0] + 88LL) + 88LL) |= 4u;
  *(_BYTE *)(*(_QWORD *)(v4 + 88) + 193LL) |= 0x10u;
  dword_4F04C3C = v1;
  return sub_729730(v8);
}
