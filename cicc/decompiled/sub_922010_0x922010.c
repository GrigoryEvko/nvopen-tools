// Function: sub_922010
// Address: 0x922010
//
__int64 __fastcall sub_922010(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  char v5; // al
  int v6; // ecx
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int8 v9; // al
  int v10; // r8d
  __int64 v11; // rax
  int v12; // r9d
  __int64 v13; // rbx
  unsigned int *v14; // r15
  unsigned int *v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rsi
  char v18; // al
  int v19; // ebx
  _BOOL4 v20; // edx
  int v22; // [rsp+0h] [rbp-80h]
  int v23; // [rsp+4h] [rbp-7Ch]
  int v24; // [rsp+4h] [rbp-7Ch]
  int v25; // [rsp+4h] [rbp-7Ch]
  unsigned __int64 v27; // [rsp+18h] [rbp-68h]
  _QWORD v28[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v29; // [rsp+40h] [rbp-40h]

  v28[0] = "temp";
  v29 = 259;
  v27 = sub_921D70(a2, a3, (__int64)v28, a4);
  if ( (*(_BYTE *)(a3 + 140) & 0xFB) != 8 || (v5 = sub_8D4C10(a3, dword_4F077C4 != 2), v6 = 1, (v5 & 2) == 0) )
  {
    v6 = unk_4D0463C;
    if ( unk_4D0463C )
      v6 = sub_90AA40(*(_QWORD *)(a2 + 32), v27);
  }
  if ( *(char *)(a3 + 142) >= 0 && *(_BYTE *)(a3 + 140) == 12 )
  {
    v25 = v6;
    LODWORD(v7) = sub_8D4AB0(a3);
    v6 = v25;
    v7 = (unsigned int)v7;
  }
  else
  {
    v7 = *(unsigned int *)(a3 + 136);
  }
  if ( v7 )
  {
    _BitScanReverse64(&v7, v7);
    v10 = (unsigned __int8)(63 - (v7 ^ 0x3F));
  }
  else
  {
    v23 = v6;
    v8 = sub_AA4E30(*(_QWORD *)(a2 + 96));
    v9 = sub_AE5020(v8, *(_QWORD *)(a4 + 8));
    v6 = v23;
    v10 = v9;
  }
  v24 = v6;
  v22 = v10;
  v29 = 257;
  v11 = sub_BD2C40(80, unk_3F10A10);
  v13 = v11;
  if ( v11 )
    sub_B4D3C0(v11, a4, v27, v24, v22, v12, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
    *(_QWORD *)(a2 + 136),
    v13,
    v28,
    *(_QWORD *)(a2 + 104),
    *(_QWORD *)(a2 + 112));
  v14 = *(unsigned int **)(a2 + 48);
  v15 = &v14[4 * *(unsigned int *)(a2 + 56)];
  while ( v15 != v14 )
  {
    v16 = *((_QWORD *)v14 + 1);
    v17 = *v14;
    v14 += 4;
    sub_B99FD0(v13, v17, v16);
  }
  v18 = *(_BYTE *)(a3 + 140);
  if ( *(char *)(a3 + 142) >= 0 && v18 == 12 )
  {
    v19 = sub_8D4AB0(a3);
    v18 = *(_BYTE *)(a3 + 140);
  }
  else
  {
    v19 = *(_DWORD *)(a3 + 136);
  }
  v20 = 0;
  if ( (v18 & 0xFB) == 8 )
    v20 = (sub_8D4C10(a3, dword_4F077C4 != 2) & 2) != 0;
  *(_QWORD *)(a1 + 16) = a3;
  *(_DWORD *)(a1 + 24) = v19;
  *(_QWORD *)(a1 + 8) = v27;
  *(_DWORD *)a1 = 0;
  *(_DWORD *)(a1 + 48) = v20;
  return a1;
}
