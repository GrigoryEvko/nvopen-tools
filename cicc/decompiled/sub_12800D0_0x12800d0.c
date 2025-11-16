// Function: sub_12800D0
// Address: 0x12800d0
//
__int64 __fastcall sub_12800D0(__int64 a1, _QWORD *a2, unsigned __int64 a3, __int64 a4)
{
  char v7; // al
  int v8; // ecx
  __int64 v9; // rax
  _QWORD *v10; // r14
  __int64 v11; // rdi
  unsigned __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // rsi
  char v17; // al
  int v18; // ebx
  _BOOL4 v19; // edx
  unsigned int v21; // [rsp+Ch] [rbp-74h]
  unsigned __int64 *v23; // [rsp+10h] [rbp-70h]
  _QWORD *v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v26[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v27; // [rsp+40h] [rbp-40h]

  v26[0] = "temp";
  v27 = 259;
  v24 = sub_127FE40(a2, a3, (__int64)v26);
  if ( (*(_BYTE *)(a3 + 140) & 0xFB) != 8 || (v7 = sub_8D4C10(a3, dword_4F077C4 != 2), v8 = 1, (v7 & 2) == 0) )
  {
    v8 = unk_4D0463C;
    if ( unk_4D0463C )
      v8 = sub_126A420(a2[4], (unsigned __int64)v24);
  }
  v21 = v8;
  v27 = 257;
  v9 = sub_1648A60(64, 2);
  v10 = (_QWORD *)v9;
  if ( v9 )
    sub_15F9650(v9, a4, v24, v21, 0);
  v11 = a2[7];
  if ( v11 )
  {
    v23 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v11 + 40, v10);
    v12 = *v23;
    v13 = v10[3] & 7LL;
    v10[4] = v23;
    v12 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v12 | v13;
    *(_QWORD *)(v12 + 8) = v10 + 3;
    *v23 = *v23 & 7 | (unsigned __int64)(v10 + 3);
  }
  sub_164B780(v10, v26);
  v14 = a2[6];
  if ( v14 )
  {
    v25 = a2[6];
    sub_1623A60(&v25, v14, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v15 = v25;
    v10[6] = v25;
    if ( v15 )
      sub_1623210(&v25, v15, v10 + 6);
  }
  if ( *(char *)(a3 + 142) >= 0 && *(_BYTE *)(a3 + 140) == 12 )
    v16 = (unsigned int)sub_8D4AB0(a3);
  else
    v16 = *(unsigned int *)(a3 + 136);
  sub_15F9450(v10, v16);
  if ( *(char *)(a3 + 142) < 0 )
  {
    v18 = *(_DWORD *)(a3 + 136);
    v19 = 0;
    if ( (*(_BYTE *)(a3 + 140) & 0xFB) != 8 )
      goto LABEL_19;
LABEL_23:
    v19 = (sub_8D4C10(a3, dword_4F077C4 != 2) & 2) != 0;
    goto LABEL_19;
  }
  v17 = *(_BYTE *)(a3 + 140);
  if ( v17 == 12 )
  {
    v18 = sub_8D4AB0(a3);
    v17 = *(_BYTE *)(a3 + 140);
  }
  else
  {
    v18 = *(_DWORD *)(a3 + 136);
  }
  v19 = 0;
  if ( (v17 & 0xFB) == 8 )
    goto LABEL_23;
LABEL_19:
  *(_DWORD *)(a1 + 16) = v18;
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = v24;
  *(_DWORD *)(a1 + 40) = v19;
  return a1;
}
