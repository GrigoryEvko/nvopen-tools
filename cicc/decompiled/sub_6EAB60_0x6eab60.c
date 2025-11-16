// Function: sub_6EAB60
// Address: 0x6eab60
//
__int64 __fastcall sub_6EAB60(__int64 a1, int a2, unsigned int a3, _DWORD *a4, _QWORD *a5, __int64 a6, __int64 a7)
{
  _QWORD *v8; // rcx
  __int64 v9; // r12
  char v11; // al
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 result; // rax
  char v20; // al
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rsi
  int v26; // eax
  __int64 v27; // rsi
  _QWORD *v28; // [rsp+8h] [rbp-48h]
  _QWORD *v29; // [rsp+8h] [rbp-48h]
  _QWORD *v30; // [rsp+8h] [rbp-48h]

  v8 = a5;
  v9 = a1;
  v11 = *(_BYTE *)(a1 + 80);
  if ( v11 == 16 )
  {
    v9 = **(_QWORD **)(a1 + 88);
    v11 = *(_BYTE *)(v9 + 80);
  }
  if ( v11 == 24 )
    v9 = *(_QWORD *)(v9 + 88);
  v12 = *(_QWORD *)(v9 + 88);
  if ( (*(_BYTE *)(v12 + 193) & 4) != 0 )
    sub_6DEC10(*(_QWORD *)(v9 + 88));
  v13 = qword_4D03C50;
  if ( dword_4F077C4 == 2 && (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 && unk_4F07290 == v12 && !dword_4F077BC )
  {
    v29 = v8;
    v26 = sub_6E5430();
    v8 = v29;
    if ( v26 )
    {
      sub_6851C0(0x186u, a4);
      v8 = v29;
    }
    v13 = qword_4D03C50;
  }
  if ( *(char *)(v13 + 18) >= 0 )
    goto LABEL_9;
  if ( (*(_BYTE *)(v9 + 104) & 1) != 0 )
  {
    v30 = v8;
    v22 = sub_8796F0(v9);
    v8 = v30;
  }
  else
  {
    v21 = *(_QWORD *)(v9 + 88);
    if ( *(_BYTE *)(v9 + 80) == 20 )
      v21 = *(_QWORD *)(v21 + 176);
    v22 = (*(_BYTE *)(v21 + 208) & 4) != 0;
  }
  if ( v22 )
    goto LABEL_35;
  if ( (*(_BYTE *)(v12 + 207) & 0x30) != 0x10 || (*(_DWORD *)(v12 + 192) & 0x8002000) != 0 )
    goto LABEL_9;
  v23 = *(_QWORD *)(v9 + 96);
  if ( !v23 )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(v9 + 88) + 208LL) & 2) != 0 )
      goto LABEL_9;
LABEL_35:
    sub_6E50A0();
    return sub_6E6260((_QWORD *)a7);
  }
  v24 = *(_QWORD *)(v23 + 32);
  if ( (unsigned __int8)(*(_BYTE *)(v24 + 80) - 19) <= 3u )
  {
    v25 = *(_QWORD *)(v24 + 88);
    if ( *(_QWORD *)(v25 + 88) )
    {
      if ( (*(_BYTE *)(v25 + 160) & 1) == 0 )
        v24 = *(_QWORD *)(v25 + 88);
    }
  }
  if ( (*(_BYTE *)(v24 + 81) & 2) == 0 )
    goto LABEL_35;
LABEL_9:
  v28 = v8;
  v14 = (__int64 *)sub_731280(v12);
  *(__int64 *)((char *)v14 + 28) = *(_QWORD *)a4;
  sub_6E7150(v14, a7);
  v16 = (__int64)v28;
  if ( (*(_BYTE *)(v12 + 89) & 4) != 0 )
  {
    v20 = (a2 == 0) & (*(_BYTE *)(v12 + 192) >> 1);
    if ( v20 )
    {
      if ( (*(_BYTE *)(v12 + 192) & 0x10) != 0 )
      {
        v20 = 0;
      }
      else
      {
        v27 = *(_QWORD *)(*(_QWORD *)(v12 + 40) + 32LL);
        if ( (unsigned __int8)(*(_BYTE *)(v27 + 140) - 9) <= 2u )
          v20 = (*(_BYTE *)(v27 + 176) & 1) == 0;
      }
    }
    *(_BYTE *)(a7 + 18) = *(_BYTE *)(a7 + 18) & 0xFB | (4 * (v20 & 1));
  }
  v17 = a3;
  v18 = *(_BYTE *)(a7 + 18) & 0xBF;
  *(_BYTE *)(a7 + 18) = *(_BYTE *)(a7 + 18) & 0xBF | ((a2 & 1) << 6);
  *(_QWORD *)(a7 + 68) = *(_QWORD *)a4;
  *(_QWORD *)(a7 + 76) = *v28;
  if ( !a3 )
  {
    v18 = 0;
    sub_6E3280(a7, 0);
  }
  *(_QWORD *)(a7 + 88) = a6;
  if ( (*(_BYTE *)(a7 + 18) & 4) == 0 )
    return (__int64)sub_6E1D20((__int64 *)v12, v18, v17, v16, v15);
  result = dword_4F077BC;
  if ( dword_4F077BC )
  {
    if ( *(char *)(v12 + 192) < 0 || (*(_BYTE *)(v12 + 195) & 1) != 0 && (result = sub_736960(v12), (_DWORD)result) )
    {
      result = qword_4D03C50;
      if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
        return sub_8AD0D0(v9, 1, 0);
    }
  }
  return result;
}
