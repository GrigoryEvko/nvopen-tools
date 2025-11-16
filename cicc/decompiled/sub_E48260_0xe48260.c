// Function: sub_E48260
// Address: 0xe48260
//
__int64 __fastcall sub_E48260(__int64 a1, bool *a2, __int64 a3, __int64 a4)
{
  char v4; // al
  char v6; // dl
  bool v8; // dl
  unsigned int v9; // eax
  unsigned int v10; // r15d
  unsigned __int8 v11; // al
  unsigned int v12; // ecx
  char v13; // al
  __int128 v14; // rax
  __int64 v15; // r13
  bool v17; // r9
  bool v18; // al
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // r13
  char v22; // bl
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rbx
  char v26; // r12
  __int128 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-F0h]
  bool v29; // [rsp+8h] [rbp-E8h]
  __int64 v30; // [rsp+8h] [rbp-E8h]
  _BYTE v31[32]; // [rsp+10h] [rbp-E0h] BYREF
  __int128 v32; // [rsp+30h] [rbp-C0h] BYREF
  const char *v33; // [rsp+40h] [rbp-B0h]
  __int64 v34; // [rsp+48h] [rbp-A8h]
  __int64 v35; // [rsp+50h] [rbp-A0h]
  const char *v36; // [rsp+60h] [rbp-90h]
  __int64 v37; // [rsp+68h] [rbp-88h]
  char v38; // [rsp+80h] [rbp-70h]
  char v39; // [rsp+81h] [rbp-6Fh]
  _OWORD v40[2]; // [rsp+90h] [rbp-60h] BYREF
  __int64 v41; // [rsp+B0h] [rbp-40h]

  v4 = *(_BYTE *)(a4 + 32) & 0xF;
  if ( v4 == 6 )
    goto LABEL_19;
  v6 = *(_BYTE *)(a3 + 32) & 0xF;
  if ( v6 == 6 )
    goto LABEL_19;
  if ( v4 == 1 )
  {
    if ( v6 != 1 )
    {
      LOBYTE(v10) = sub_B2FC80(a3);
      goto LABEL_17;
    }
LABEL_22:
    if ( (*(_BYTE *)(a4 + 33) & 3) != 1 )
      goto LABEL_23;
    LOBYTE(v10) = 1;
LABEL_30:
    *a2 = v10;
    return 0;
  }
  v8 = sub_B2FC80(a4);
  if ( (*(_BYTE *)(a3 + 32) & 0xF) == 1 )
  {
    if ( !v8 )
      goto LABEL_19;
    goto LABEL_22;
  }
  v29 = v8;
  LOBYTE(v9) = sub_B2FC80(a3);
  v10 = v9;
  if ( !v29 )
  {
    if ( (_BYTE)v9 )
      goto LABEL_19;
    v11 = *(_BYTE *)(a4 + 32) & 0xF;
    if ( v11 == 10 )
    {
      v19 = *(_BYTE *)(a3 + 32) & 0xF;
      if ( ((v19 + 14) & 0xFu) <= 3 )
        goto LABEL_19;
      if ( v19 == 10 )
      {
        v20 = sub_B2F730(a3);
        v21 = *(_QWORD *)(a3 + 24);
        v30 = v20;
        v22 = sub_AE5020(v20, v21);
        v23 = sub_9208B0(v30, v21);
        *((_QWORD *)&v40[0] + 1) = v24;
        *(_QWORD *)&v40[0] = ((1LL << v22) + ((unsigned __int64)(v23 + 7) >> 3) - 1) >> v22 << v22;
        v25 = sub_CA1930(v40);
        v28 = *(_QWORD *)(a4 + 24);
        v26 = sub_AE5020(v30, v28);
        *(_QWORD *)&v27 = (((unsigned __int64)(sub_9208B0(v30, v28) + 7) >> 3) + (1LL << v26) - 1) >> v26 << v26;
        v40[0] = v27;
        *a2 = v25 < sub_CA1930(v40);
        return v10;
      }
    }
    else
    {
      v12 = v11 - 4;
      if ( v12 <= 1 || v11 == 2 )
      {
        if ( (*(_BYTE *)(a3 + 32) & 0xFu) - 2 <= 1 && v12 <= 1 )
          goto LABEL_19;
      }
      else if ( ((v11 + 7) & 0xFu) > 1 && v11 != 3 )
      {
        v13 = *(_BYTE *)(a3 + 32) & 0xF;
        if ( ((v13 + 14) & 0xFu) > 3 && ((v13 + 7) & 0xFu) > 1 )
        {
          v39 = 1;
          v36 = "': symbol multiply defined!";
          v38 = 3;
          *(_QWORD *)&v14 = sub_BD5D20(a4);
          v40[1] = v14;
          LOWORD(v41) = 1283;
          *(_QWORD *)&v40[0] = "Linking globals named '";
          v33 = "': symbol multiply defined!";
          *(_QWORD *)&v32 = v40;
          v34 = v37;
          LOWORD(v35) = 770;
          v10 = 1;
          v15 = **(_QWORD **)(a1 + 8);
          sub_1061A30(v31, 0, &v32);
          sub_B6EB20(v15, (__int64)v31);
          return v10;
        }
LABEL_19:
        *a2 = 1;
        return 0;
      }
    }
    *a2 = 0;
    return v10;
  }
LABEL_17:
  if ( (*(_BYTE *)(a4 + 33) & 3) == 1 )
    goto LABEL_30;
  if ( (*(_BYTE *)(a3 + 32) & 0xF) == 9 )
    goto LABEL_19;
LABEL_23:
  v17 = sub_B2FC80(a4);
  v18 = 0;
  if ( !v17 )
    v18 = sub_B2FC80(a3);
  *a2 = v18;
  return 0;
}
