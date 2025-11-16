// Function: sub_2C1E1C0
// Address: 0x2c1e1c0
//
__int64 *__fastcall sub_2C1E1C0(__int64 a1, __int64 a2)
{
  unsigned int **v3; // r14
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r11
  unsigned int v13; // eax
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v25; // [rsp+18h] [rbp-88h]
  __int64 v26; // [rsp+20h] [rbp-80h]
  unsigned int v27; // [rsp+28h] [rbp-78h]
  _BYTE *v28; // [rsp+28h] [rbp-78h]
  _BYTE *v29; // [rsp+30h] [rbp-70h] BYREF
  _BYTE *v30; // [rsp+38h] [rbp-68h] BYREF
  __int64 v31[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v32; // [rsp+60h] [rbp-40h]

  v3 = *(unsigned int ***)(a2 + 904);
  v31[0] = *(_QWORD *)(a1 + 88);
  if ( v31[0] )
    sub_2AAAFA0(v31);
  sub_2BF1A90(a2, (__int64)v31);
  sub_9C6650(v31);
  if ( *(_DWORD *)(a1 + 56) == 3 && (v4 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL)) != 0 )
  {
    v5 = *(_QWORD *)(v4 + 40);
    if ( *(_DWORD *)(v5 + 32) <= 0x40u )
      v6 = *(_QWORD *)(v5 + 24);
    else
      v6 = **(_QWORD **)(v5 + 24);
    v7 = v6;
    v26 = -(__int64)(unsigned int)v6;
  }
  else
  {
    v26 = 0;
    v7 = 0;
  }
  v8 = sub_2C0D490(*(_BYTE *)(a2 + 12), 1, v7, (__int64)v3);
  BYTE4(v31[0]) = 0;
  v9 = v8;
  v10 = *(_QWORD *)(a1 + 48);
  LODWORD(v31[0]) = 0;
  v11 = sub_2BFB120(a2, *(_QWORD *)(v10 + 8), (unsigned int *)v31);
  v12 = v11;
  if ( v9 != *(_QWORD *)(v11 + 8) )
  {
    v25 = v11;
    v32 = 257;
    v27 = sub_BCB060(*(_QWORD *)(v11 + 8));
    v13 = sub_BCB060(v9);
    v12 = v25;
    if ( v27 < v13 )
    {
      v12 = sub_A82F30(v3, v25, v9, (__int64)v31, 0);
    }
    else if ( v27 > v13 )
    {
      v12 = sub_A82DA0(v3, v25, v9, (__int64)v31, 0, 0);
    }
  }
  v28 = (_BYTE *)v12;
  v32 = 257;
  v14 = (_BYTE *)sub_AD64C0(v9, v26, 0);
  v15 = (_BYTE *)sub_A81850(v3, v14, v28, (__int64)v31, 0, 0);
  v32 = 257;
  v29 = v15;
  v16 = (_BYTE *)sub_AD64C0(v9, 1, 0);
  v17 = (_BYTE *)sub_929DE0(v3, v16, v28, (__int64)v31, 0, 0);
  BYTE4(v31[0]) = 0;
  v30 = v17;
  v18 = *(__int64 **)(a1 + 48);
  LODWORD(v31[0]) = 0;
  v19 = sub_2BFB120(a2, *v18, (unsigned int *)v31);
  v20 = *(_DWORD *)(a1 + 156);
  v21 = *(_QWORD *)(a1 + 160);
  v32 = 257;
  v22 = sub_921130(v3, v21, v19, &v29, 1, (__int64)v31, v20);
  v32 = 257;
  v23 = sub_921130(v3, *(_QWORD *)(a1 + 160), v22, &v30, 1, (__int64)v31, *(_DWORD *)(a1 + 156));
  return sub_2BF26E0(a2, a1 + 96, v23, 1);
}
