// Function: sub_1F39950
// Address: 0x1f39950
//
_QWORD *__fastcall sub_1F39950(__int64 a1, __int64 a2, char a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // r9
  unsigned int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rsi
  int v10; // r8d
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 v13; // rbx
  __int64 v14; // r10
  int v15; // r13d
  unsigned __int8 v16; // al
  char v17; // r10
  char v18; // r11
  unsigned __int8 v19; // al
  unsigned __int8 v20; // al
  __int64 v21; // rax
  int v22; // eax
  int v23; // r15d
  int v24; // r14d
  _QWORD *v25; // rax
  _QWORD *result; // rax
  __int64 v27; // rax
  int v28; // eax
  __int64 v29; // rax
  char v30; // [rsp+4h] [rbp-7Ch]
  char v31; // [rsp+8h] [rbp-78h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  __int64 v33; // [rsp+10h] [rbp-70h]
  int v34; // [rsp+18h] [rbp-68h]
  __int64 v35; // [rsp+20h] [rbp-60h]
  __int64 v36; // [rsp+20h] [rbp-60h]
  int v37; // [rsp+28h] [rbp-58h]
  int v38; // [rsp+2Ch] [rbp-54h]
  unsigned __int8 v39; // [rsp+2Ch] [rbp-54h]
  char v40; // [rsp+32h] [rbp-4Eh]
  unsigned __int8 v41; // [rsp+33h] [rbp-4Dh]
  unsigned __int16 v42; // [rsp+34h] [rbp-4Ch]
  int v43; // [rsp+34h] [rbp-4Ch]
  unsigned __int16 v44; // [rsp+38h] [rbp-48h]
  char v45; // [rsp+3Ah] [rbp-46h]
  bool v46; // [rsp+3Bh] [rbp-45h]
  bool v47; // [rsp+3Ch] [rbp-44h]
  char v48; // [rsp+3Dh] [rbp-43h]
  char v49; // [rsp+3Eh] [rbp-42h]
  char v50; // [rsp+3Fh] [rbp-41h]
  __int64 v51; // [rsp+40h] [rbp-40h]
  _QWORD *v52; // [rsp+40h] [rbp-40h]
  __int64 v53; // [rsp+40h] [rbp-40h]
  int v54; // [rsp+48h] [rbp-38h]
  _QWORD *v55; // [rsp+48h] [rbp-38h]

  v5 = a2;
  v7 = a4;
  v40 = *(_BYTE *)(*(_QWORD *)(a2 + 16) + 4LL);
  v8 = *(_QWORD *)(a2 + 32);
  if ( v40 )
  {
    if ( *(_BYTE *)v8 )
      return 0;
    a4 = (unsigned int)a4;
    v9 = a5;
    v10 = *(_DWORD *)(v8 + 8);
    v11 = 40LL * (unsigned int)a4;
    v12 = v8 + v11;
    v54 = *(_DWORD *)(v8 + v11 + 8);
    v13 = 40LL * a5;
    v14 = v8 + v13;
    v15 = *(_DWORD *)(v8 + v13 + 8);
    v37 = (*(_DWORD *)v8 >> 8) & 0xFFF;
  }
  else
  {
    a4 = (unsigned int)a4;
    v9 = a5;
    v37 = 0;
    v10 = 0;
    v11 = 40LL * (unsigned int)a4;
    v12 = v8 + v11;
    v54 = *(_DWORD *)(v8 + v11 + 8);
    v13 = 40 * v9;
    v14 = v8 + 40 * v9;
    v15 = *(_DWORD *)(v14 + 8);
  }
  v41 = 0;
  v44 = (*(_DWORD *)v12 >> 8) & 0xFFF;
  v42 = (*(_DWORD *)v14 >> 8) & 0xFFF;
  v16 = *(_BYTE *)(v14 + 3);
  v17 = *(_BYTE *)(v14 + 4);
  v49 = ((*(_BYTE *)(v12 + 3) & 0x40) != 0) & ((*(_BYTE *)(v12 + 3) >> 4) ^ 1);
  v50 = ((v16 & 0x40) != 0) & ((v16 >> 4) ^ 1);
  v18 = *(_BYTE *)(v12 + 4);
  v45 = v18 & 1;
  v47 = (v18 & 2) != 0;
  v46 = (v17 & 2) != 0;
  v48 = v17 & 1;
  if ( v54 > 0 )
  {
    v31 = a3;
    v32 = v5;
    v35 = a4;
    v38 = v10;
    v19 = sub_1E31310(v12);
    a3 = v31;
    v5 = v32;
    a4 = v35;
    v41 = v19;
    v10 = v38;
  }
  v39 = 0;
  if ( v15 > 0 )
  {
    v30 = a3;
    v33 = a4;
    v34 = v10;
    v36 = v5;
    v20 = sub_1E31310(v13 + *(_QWORD *)(v5 + 32));
    a3 = v30;
    a4 = v33;
    v10 = v34;
    v39 = v20;
    v5 = v36;
  }
  if ( v10 == v54 )
  {
    if ( v40 )
    {
      v21 = *(_QWORD *)(v5 + 16);
      if ( v7 < *(unsigned __int16 *)(v21 + 2) )
      {
        v22 = *(_DWORD *)(*(_QWORD *)(v21 + 40) + 8 * a4 + 4);
        if ( (v22 & 1) != 0 && (v22 & 0xFF0000) == 0 )
        {
          v23 = v44;
          v50 = 0;
          v10 = v15;
          v37 = v42;
          v24 = v42;
          goto LABEL_14;
        }
      }
    }
  }
  if ( v10 != v15 || !v40 )
  {
    v23 = v44;
    v24 = v42;
    v25 = (_QWORD *)v5;
    if ( !a3 )
      goto LABEL_30;
    goto LABEL_27;
  }
  v27 = *(_QWORD *)(v5 + 16);
  if ( a5 >= *(unsigned __int16 *)(v27 + 2)
    || (v28 = *(_DWORD *)(*(_QWORD *)(v27 + 40) + 8 * v9 + 4), (v28 & 1) == 0)
    || (v28 & 0xFF0000) != 0 )
  {
    v23 = v44;
    v24 = v42;
LABEL_14:
    if ( !a3 )
      goto LABEL_15;
    goto LABEL_27;
  }
  v10 = v54;
  v49 = 0;
  v24 = v42;
  v37 = v44;
  v23 = v44;
  if ( !a3 )
    goto LABEL_15;
LABEL_27:
  v43 = v10;
  v53 = v5;
  v29 = sub_1E15F70(v5);
  v25 = sub_1E0B7C0(v29, v53);
  v10 = v43;
LABEL_30:
  if ( !v40 )
    goto LABEL_16;
  v5 = (__int64)v25;
LABEL_15:
  v51 = v5;
  sub_1E310D0(*(_QWORD *)(v5 + 32), v10);
  **(_DWORD **)(v51 + 32) = (v37 << 8) | **(_DWORD **)(v51 + 32) & 0xFFF000FF;
  v25 = (_QWORD *)v51;
LABEL_16:
  v52 = v25;
  sub_1E310D0(v13 + v25[4], v54);
  sub_1E310D0(v52[4] + v11, v15);
  result = v52;
  *(_DWORD *)(v52[4] + v13) = *(_DWORD *)(v52[4] + v13) & 0xFFF000FF | (v23 << 8);
  *(_DWORD *)(v52[4] + v11) = *(_DWORD *)(v52[4] + v11) & 0xFFF000FF | (v24 << 8);
  *(_BYTE *)(v52[4] + v13 + 3) = (v49 << 6) | *(_BYTE *)(v52[4] + v13 + 3) & 0xBF;
  *(_BYTE *)(v52[4] + v11 + 3) = (v50 << 6) | *(_BYTE *)(v52[4] + v11 + 3) & 0xBF;
  *(_BYTE *)(v52[4] + v13 + 4) = v45 | *(_BYTE *)(v52[4] + v13 + 4) & 0xFE;
  *(_BYTE *)(v52[4] + v11 + 4) = v48 | *(_BYTE *)(v52[4] + v11 + 4) & 0xFE;
  *(_BYTE *)(v52[4] + v13 + 4) = (2 * v47) | *(_BYTE *)(v52[4] + v13 + 4) & 0xFD;
  *(_BYTE *)(v52[4] + v11 + 4) = (2 * v46) | *(_BYTE *)(v52[4] + v11 + 4) & 0xFD;
  if ( v54 > 0 )
  {
    sub_1E31360(v52[4] + v13, v41);
    result = v52;
  }
  if ( v15 > 0 )
  {
    v55 = result;
    sub_1E31360(v11 + result[4], v39);
    return v55;
  }
  return result;
}
