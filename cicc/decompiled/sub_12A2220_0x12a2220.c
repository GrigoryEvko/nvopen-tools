// Function: sub_12A2220
// Address: 0x12a2220
//
__int64 __fastcall sub_12A2220(__int64 a1, __int64 a2, __int64 a3, int *a4, __int64 a5)
{
  char *v6; // r12
  const char *v7; // r14
  int v8; // r15d
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r11
  unsigned int v12; // r9d
  char v13; // r8
  int v14; // ecx
  int v15; // r10d
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // r11
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 result; // rax
  int v22; // ecx
  int v23; // eax
  __int64 v24; // rax
  unsigned int v25; // [rsp+0h] [rbp-70h]
  bool v26; // [rsp+4h] [rbp-6Ch]
  unsigned int v27; // [rsp+4h] [rbp-6Ch]
  __int64 v28; // [rsp+8h] [rbp-68h]
  bool v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  int v32; // [rsp+18h] [rbp-58h]
  __int64 v33; // [rsp+18h] [rbp-58h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 v37; // [rsp+20h] [rbp-50h]
  unsigned int v39; // [rsp+34h] [rbp-3Ch] BYREF
  _QWORD v40[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = *(char **)(a2 + 8);
  if ( !v6 )
  {
    v6 = "this";
    if ( (*(_BYTE *)(a2 + 172) & 1) == 0 )
      v6 = (char *)byte_3F871B3;
  }
  sub_129E300(*(_DWORD *)(a2 + 64), (char *)&v39);
  v7 = (const char *)sub_129E180((__int64)v6, a2);
  v8 = sub_129F850(a1, *(_DWORD *)(a2 + 64));
  v9 = sub_12A0C10(a1, *(_QWORD *)(a2 + 120));
  v10 = *(_QWORD *)(a1 + 544);
  if ( v10 == *(_QWORD *)(a1 + 552) )
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 568) - 8LL) + 512LL;
  v11 = *(_QWORD *)(v10 - 8);
  v12 = v39;
  v13 = unk_4D04660 != 0;
  if ( *((_BYTE *)a4 + 4) )
  {
    v14 = *a4;
    v15 = 0;
    if ( v7 )
    {
      v25 = v39;
      v26 = unk_4D04660 != 0;
      v28 = *(_QWORD *)(v10 - 8);
      v32 = *a4;
      v16 = strlen(v7);
      v12 = v25;
      v13 = v26;
      v11 = v28;
      v15 = v16;
      v14 = v32;
    }
    v33 = v11;
    v17 = sub_15A7E40((int)a1 + 16, v11, (_DWORD)v7, v15, v14, v8, v12, v9, v13, 0);
    v18 = v33;
    v19 = v17;
  }
  else
  {
    v22 = 0;
    if ( v7 )
    {
      v27 = v39;
      v29 = unk_4D04660 != 0;
      v30 = *(_QWORD *)(v10 - 8);
      v23 = strlen(v7);
      v12 = v27;
      v13 = v29;
      v11 = v30;
      v22 = v23;
    }
    v35 = v11;
    v24 = sub_15A7DF0((int)a1 + 16, v11, (_DWORD)v7, v22, v8, v12, v9, v13, 0, 0, 0);
    v18 = v35;
    v19 = v24;
  }
  v34 = *(_QWORD *)(a5 + 8);
  sub_15C7110(v40, v39, *(unsigned __int16 *)(a2 + 68), v18, 0);
  v37 = sub_15C70A0(v40);
  v20 = sub_15A6870(a1 + 16, 0, 0);
  sub_15A7520(a1 + 16, a3, v19, v20, v37, v34);
  if ( v40[0] )
    sub_161E7C0(v40);
  *(_DWORD *)(a1 + 488) = *(_DWORD *)(a2 + 64);
  result = *(unsigned __int16 *)(a2 + 68);
  *(_WORD *)(a1 + 492) = result;
  return result;
}
