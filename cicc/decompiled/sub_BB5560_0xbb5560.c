// Function: sub_BB5560
// Address: 0xbb5560
//
__int64 __fastcall sub_BB5560(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rbx
  signed __int64 v4; // r15
  unsigned int v5; // r12d
  _BYTE *v6; // r9
  unsigned __int64 v7; // r13
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // r15
  __int64 v16; // rax
  unsigned __int8 v17; // dl
  __int64 v19; // r13
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // r10
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // r13
  unsigned __int64 v29; // r8
  __int64 v30; // [rsp+0h] [rbp-60h]
  _BYTE *v31; // [rsp+8h] [rbp-58h]
  char v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  unsigned __int64 v35; // [rsp+20h] [rbp-40h] BYREF
  __int64 v36; // [rsp+28h] [rbp-38h]

  v34 = a1;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v2 = *(_QWORD *)(a1 - 8);
  else
    v2 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v3 = (_QWORD *)(v2 + 32);
  v4 = sub_BB5290(a1) & 0xFFFFFFFFFFFFFFF9LL | 4;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v34 = *(_QWORD *)(a1 - 8) + 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v5 = 32;
  if ( v3 != (_QWORD *)v34 )
  {
    while ( 1 )
    {
      v6 = (_BYTE *)*v3;
      if ( *(_BYTE *)*v3 == 17 )
        break;
      v20 = (v4 >> 1) & 3;
      if ( !v4 )
      {
        v22 = 0;
        v19 = 1;
        goto LABEL_50;
      }
      if ( v20 )
      {
        v19 = 1;
        goto LABEL_29;
      }
      v7 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v6 = 0;
LABEL_13:
        v31 = v6;
        v8 = sub_AE4AC0(a2, v7);
        v9 = (_QWORD *)*((_QWORD *)v31 + 3);
        if ( *((_DWORD *)v31 + 8) > 0x40u )
          v9 = (_QWORD *)*v9;
        v10 = v8 + 16LL * (unsigned int)v9 + 24;
        v11 = *(_QWORD *)v10;
        LOBYTE(v10) = *(_BYTE *)(v10 + 8);
        v35 = v11;
        LOBYTE(v36) = v10;
        v12 = sub_CA1930(&v35);
        v13 = (v12 | (1LL << v5)) & -(v12 | (1LL << v5));
        if ( !v13 )
        {
          v5 = -1;
          if ( ((v4 >> 1) & 3) == 1 )
            goto LABEL_40;
          goto LABEL_25;
        }
LABEL_16:
        _BitScanReverse64(&v13, v13);
        v5 = 63 - (v13 ^ 0x3F);
        goto LABEL_17;
      }
      v19 = 1;
      v22 = 0;
LABEL_53:
      v20 = (v4 >> 1) & 3;
LABEL_50:
      v33 = v20;
      v22 = sub_BCBAE0(v22, *v3);
      if ( v33 != 1 )
        goto LABEL_31;
LABEL_45:
      v27 = sub_9208B0(a2, v22);
      v36 = v24;
      v25 = (unsigned __int64)(v27 + 7) >> 3;
LABEL_32:
      LOBYTE(v36) = v24;
      v35 = v25 * v19;
      v26 = sub_CA1930(&v35) | (1LL << v5);
      v5 = -1;
      v13 = v26 & -v26;
      if ( v13 )
        goto LABEL_16;
LABEL_17:
      if ( !v4 )
      {
        v7 = 0;
LABEL_25:
        v7 = sub_BCBAE0(v7, *v3);
        goto LABEL_20;
      }
      v14 = v4;
      v15 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      v7 = v15;
      v16 = (v14 >> 1) & 3;
      if ( v16 != 2 )
      {
        if ( v16 == 1 && v15 )
        {
LABEL_40:
          v7 = *(_QWORD *)(v7 + 24);
          goto LABEL_20;
        }
        goto LABEL_25;
      }
      if ( !v15 )
        goto LABEL_25;
LABEL_20:
      v17 = *(_BYTE *)(v7 + 8);
      if ( v17 == 16 )
      {
        v4 = *(_QWORD *)(v7 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
LABEL_8:
        v3 += 4;
        if ( (_QWORD *)v34 == v3 )
          return v5;
      }
      else
      {
        if ( (unsigned int)v17 - 17 > 1 )
        {
          v28 = v7 & 0xFFFFFFFFFFFFFFF9LL;
          v29 = 0;
          if ( v17 == 15 )
            v29 = v28;
          v4 = v29;
          goto LABEL_8;
        }
        v3 += 4;
        v4 = v7 & 0xFFFFFFFFFFFFFFF9LL | 2;
        if ( (_QWORD *)v34 == v3 )
          return v5;
      }
    }
    if ( v4 )
    {
      if ( (v4 & 6) == 0 )
      {
        v7 = v4 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_13;
      }
    }
    v19 = *((_QWORD *)v6 + 3);
    if ( *((_DWORD *)v6 + 8) > 0x40u )
      v19 = *(_QWORD *)v19;
    v20 = (v4 >> 1) & 3;
    if ( !v4 )
      goto LABEL_41;
LABEL_29:
    v21 = v4 & 0xFFFFFFFFFFFFFFF8LL;
    v22 = v4 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v20 == 2 )
    {
      if ( !v21 )
LABEL_41:
        v22 = sub_BCBAE0(0, *v3);
LABEL_31:
      v30 = v22;
      v32 = sub_AE5020(a2, v22);
      v23 = sub_9208B0(a2, v30);
      v36 = v24;
      v25 = ((1LL << v32) + ((unsigned __int64)(v23 + 7) >> 3) - 1) >> v32 << v32;
      goto LABEL_32;
    }
    if ( v20 == 1 && v21 )
    {
      v22 = *(_QWORD *)(v21 + 24);
      goto LABEL_45;
    }
    goto LABEL_53;
  }
  return v5;
}
