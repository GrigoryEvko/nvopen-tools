// Function: sub_AFE760
// Address: 0xafe760
//
__int64 __fastcall sub_AFE760(__int64 a1, _BYTE **a2, _QWORD *a3)
{
  int v4; // r14d
  _BYTE *v6; // rbx
  __int64 v7; // r15
  __int64 v9; // rcx
  unsigned __int8 v10; // al
  unsigned __int8 v11; // al
  unsigned __int8 v12; // al
  __int64 v13; // rcx
  int v14; // r14d
  int v15; // eax
  _BYTE *v16; // rsi
  int v17; // r8d
  _QWORD *v18; // rdi
  unsigned int v19; // eax
  _QWORD *v20; // rcx
  _BYTE *v21; // rdx
  _BYTE *v22; // [rsp+8h] [rbp-78h]
  __int64 v23; // [rsp+18h] [rbp-68h] BYREF
  __int64 v24; // [rsp+20h] [rbp-60h] BYREF
  __int64 v25; // [rsp+28h] [rbp-58h] BYREF
  __int64 v26[2]; // [rsp+30h] [rbp-50h] BYREF
  int v27; // [rsp+40h] [rbp-40h]
  char v28; // [rsp+44h] [rbp-3Ch]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 8);
  v9 = (__int64)(*a2 - 16);
  if ( **a2 != 16 )
  {
    v22 = *a2 - 16;
    sub_A17150(v22);
    v9 = (__int64)v22;
  }
  v10 = *(v6 - 16);
  if ( (v10 & 2) != 0 )
  {
    v23 = *(_QWORD *)(*((_QWORD *)v6 - 4) + 8LL);
    v11 = *(v6 - 16);
    if ( (v11 & 2) != 0 )
      goto LABEL_7;
LABEL_16:
    v24 = *(_QWORD *)(v9 - 8LL * ((v11 >> 2) & 0xF) + 16);
    v12 = *(v6 - 16);
    if ( (v12 & 2) != 0 )
      goto LABEL_8;
    goto LABEL_17;
  }
  v23 = *(_QWORD *)(v9 - 8LL * ((v10 >> 2) & 0xF) + 8);
  v11 = *(v6 - 16);
  if ( (v11 & 2) == 0 )
    goto LABEL_16;
LABEL_7:
  v24 = *(_QWORD *)(*((_QWORD *)v6 - 4) + 16LL);
  v12 = *(v6 - 16);
  if ( (v12 & 2) != 0 )
  {
LABEL_8:
    v13 = *((_QWORD *)v6 - 4);
    goto LABEL_9;
  }
LABEL_17:
  v13 = v9 - 8LL * ((v12 >> 2) & 0xF);
LABEL_9:
  v14 = v4 - 1;
  v25 = *(_QWORD *)(v13 + 24);
  v26[0] = sub_AF5140((__int64)v6, 4u);
  v26[1] = sub_AF5140((__int64)v6, 5u);
  v27 = *((_DWORD *)v6 + 1);
  v28 = v6[1] >> 7;
  v15 = sub_AFBE30(&v23, &v24, &v25, v26);
  v16 = *a2;
  v17 = 1;
  v18 = 0;
  v19 = v14 & v15;
  v20 = (_QWORD *)(v7 + 8LL * v19);
  v21 = (_BYTE *)*v20;
  if ( (_BYTE *)*v20 == *a2 )
  {
LABEL_19:
    *a3 = v20;
    return 1;
  }
  else
  {
    while ( v21 != (_BYTE *)-4096LL )
    {
      if ( v21 != (_BYTE *)-8192LL || v18 )
        v20 = v18;
      v19 = v14 & (v17 + v19);
      v21 = *(_BYTE **)(v7 + 8LL * v19);
      if ( v21 == v16 )
      {
        v20 = (_QWORD *)(v7 + 8LL * v19);
        goto LABEL_19;
      }
      ++v17;
      v18 = v20;
      v20 = (_QWORD *)(v7 + 8LL * v19);
    }
    if ( !v18 )
      v18 = v20;
    *a3 = v18;
    return 0;
  }
}
