// Function: sub_ADC010
// Address: 0xadc010
//
_QWORD *__fastcall sub_ADC010(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rbx
  unsigned __int8 v5; // al
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // r13
  unsigned __int16 v9; // dx
  unsigned int v10; // edx
  __int64 v11; // rdi
  _QWORD *v12; // rcx
  _QWORD *v13; // rdi
  __int64 *v14; // rdx
  char v15; // cl
  __int64 v16; // rsi
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // r10
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r15
  __int64 v24; // rbx
  __int64 v25; // rax
  _QWORD *v26; // r14
  __int64 v27; // rbx
  __int64 v28; // rax
  int v29; // r13d
  int v30; // eax
  int v31; // edx
  unsigned int v33; // edi
  _QWORD *v34; // rcx
  __int64 v35; // rdi
  __int64 v36; // [rsp+8h] [rbp-B8h]
  __int64 v37; // [rsp+10h] [rbp-B0h]
  __int64 v38; // [rsp+18h] [rbp-A8h]
  __int64 v39; // [rsp+20h] [rbp-A0h]
  __int64 v40; // [rsp+20h] [rbp-A0h]
  __int64 v41; // [rsp+20h] [rbp-A0h]
  __int64 v42; // [rsp+28h] [rbp-98h]
  __int64 v43; // [rsp+30h] [rbp-90h]
  __int64 v44; // [rsp+38h] [rbp-88h]
  __int64 v45; // [rsp+38h] [rbp-88h]
  __int64 v46; // [rsp+40h] [rbp-80h]
  __int64 v47; // [rsp+48h] [rbp-78h]
  int v48; // [rsp+50h] [rbp-70h]
  int v49; // [rsp+54h] [rbp-6Ch]
  __int64 v50; // [rsp+58h] [rbp-68h]
  __int64 v51; // [rsp+60h] [rbp-60h]
  int v52; // [rsp+68h] [rbp-58h]
  int v53; // [rsp+6Ch] [rbp-54h]
  int v54; // [rsp+70h] [rbp-50h]
  int v55; // [rsp+74h] [rbp-4Ch]
  __int64 v56; // [rsp+78h] [rbp-48h]
  __int64 v57; // [rsp+80h] [rbp-40h]
  __int64 v58; // [rsp+88h] [rbp-38h]

  v4 = a2 - 16;
  v5 = *(_BYTE *)(a2 - 16);
  if ( (v5 & 2) != 0 )
  {
    if ( *(_DWORD *)(a2 - 24) <= 0xCu || (v6 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 96LL)) == 0 )
    {
      v38 = 0;
      v8 = 0;
      goto LABEL_28;
    }
  }
  else
  {
    v9 = *(_WORD *)(a2 - 16);
    if ( ((v9 >> 6) & 0xFu) <= 0xC || (v6 = *(_QWORD *)(v4 - 8LL * ((v5 >> 2) & 0xF) + 96)) == 0 )
    {
      v38 = 0;
      v8 = 0;
LABEL_6:
      v10 = (v9 >> 6) & 0xF;
      v11 = 8LL * ((v5 >> 2) & 0xF);
      v12 = (_QWORD *)(v4 - v11);
      if ( v10 <= 0xB )
      {
        if ( v10 <= 0xA )
        {
          v13 = (_QWORD *)(v4 - v11);
          v57 = v12[7];
          v58 = v12[6];
          if ( v10 <= 9 )
          {
            v51 = v12[5];
            v52 = *(_DWORD *)(a2 + 36);
            v53 = *(_DWORD *)(a2 + 32);
            v54 = *(_DWORD *)(a2 + 28);
            v55 = *(_DWORD *)(a2 + 24);
            if ( v10 <= 8 )
            {
              v46 = 0;
              v56 = 0;
              v49 = *(_DWORD *)(a2 + 20);
              v50 = 0;
              v47 = 0;
LABEL_11:
              v48 = *(_DWORD *)(a2 + 16);
              v14 = (__int64 *)(v4 - 8LL * ((v5 >> 2) & 0xF));
              v15 = 0;
              v16 = v14[4];
              v42 = v16;
              if ( *(_BYTE *)a2 != 16 )
                goto LABEL_12;
              goto LABEL_35;
            }
            v46 = 0;
            v56 = 0;
            v50 = 0;
LABEL_10:
            v47 = v12[8];
            v49 = *(_DWORD *)(a2 + 20);
            goto LABEL_11;
          }
          v46 = 0;
          v56 = 0;
LABEL_9:
          v50 = v13[9];
          v55 = *(_DWORD *)(a2 + 24);
          v12 = (_QWORD *)(v4 - 8LL * ((v5 >> 2) & 0xF));
          v51 = v12[5];
          v52 = *(_DWORD *)(a2 + 36);
          v53 = *(_DWORD *)(a2 + 32);
          v54 = *(_DWORD *)(a2 + 28);
          goto LABEL_10;
        }
        v46 = 0;
      }
      else
      {
        v46 = v12[11];
      }
      v56 = v12[10];
      v13 = (_QWORD *)(v4 - 8LL * ((v5 >> 2) & 0xF));
      v57 = v13[7];
      v58 = v13[6];
      goto LABEL_9;
    }
  }
  v38 = sub_B91420(v6, a2);
  v5 = *(_BYTE *)(a2 - 16);
  v8 = v7;
  if ( (v5 & 2) == 0 )
  {
    v9 = *(_WORD *)(a2 - 16);
    goto LABEL_6;
  }
LABEL_28:
  v33 = *(_DWORD *)(a2 - 24);
  v34 = *(_QWORD **)(a2 - 32);
  if ( v33 > 0xB )
  {
    v46 = v34[11];
LABEL_30:
    v56 = v34[10];
    v57 = v34[7];
    v58 = v34[6];
    v50 = v34[9];
LABEL_31:
    v14 = *(__int64 **)(a2 - 32);
    goto LABEL_32;
  }
  if ( v33 > 0xA )
  {
    v46 = 0;
    goto LABEL_30;
  }
  v14 = *(__int64 **)(a2 - 32);
  v57 = v34[7];
  v58 = v34[6];
  if ( v33 == 10 )
  {
    v46 = 0;
    v56 = 0;
    v50 = v34[9];
    goto LABEL_31;
  }
  v46 = 0;
  v56 = 0;
  v50 = 0;
LABEL_32:
  v51 = v14[5];
  v52 = *(_DWORD *)(a2 + 36);
  v53 = *(_DWORD *)(a2 + 32);
  v54 = *(_DWORD *)(a2 + 28);
  v55 = *(_DWORD *)(a2 + 24);
  if ( v33 <= 8 )
  {
    v47 = 0;
  }
  else
  {
    v35 = v14[8];
    v14 = *(__int64 **)(a2 - 32);
    v47 = v35;
  }
  v49 = *(_DWORD *)(a2 + 20);
  v16 = v14[4];
  v15 = 1;
  v42 = v16;
  v48 = *(_DWORD *)(a2 + 16);
  if ( *(_BYTE *)a2 != 16 )
  {
LABEL_12:
    v16 = *v14;
    v43 = *v14;
    if ( v15 )
      goto LABEL_13;
    goto LABEL_36;
  }
LABEL_35:
  LODWORD(v43) = a2;
  if ( v15 )
  {
LABEL_13:
    v17 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 24LL);
    if ( v17 )
      goto LABEL_14;
    goto LABEL_37;
  }
LABEL_36:
  v17 = *(_QWORD *)(v4 - 8LL * ((v5 >> 2) & 0xF) + 24);
  if ( v17 )
  {
LABEL_14:
    v17 = sub_B91420(v17, v16);
    v5 = *(_BYTE *)(a2 - 16);
    v19 = v18;
    if ( (v5 & 2) != 0 )
      goto LABEL_15;
    goto LABEL_38;
  }
LABEL_37:
  v19 = 0;
  if ( (v5 & 2) != 0 )
  {
LABEL_15:
    v20 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    if ( !v20 )
    {
      v23 = 0;
      goto LABEL_17;
    }
    goto LABEL_16;
  }
LABEL_38:
  v20 = *(_QWORD *)(v4 - 8LL * ((v5 >> 2) & 0xF) + 16);
  if ( !v20 )
  {
    v23 = 0;
    goto LABEL_40;
  }
LABEL_16:
  v39 = v17;
  v44 = v19;
  v21 = sub_B91420(v20, v16);
  v19 = v44;
  v17 = v39;
  v20 = v21;
  v5 = *(_BYTE *)(a2 - 16);
  v23 = v22;
  if ( (v5 & 2) != 0 )
  {
LABEL_17:
    v24 = *(_QWORD *)(a2 - 32);
    goto LABEL_18;
  }
LABEL_40:
  v24 = v4 - 8LL * ((v5 >> 2) & 0xF);
LABEL_18:
  v45 = *(_QWORD *)(v24 + 8);
  v25 = *(_QWORD *)(a2 + 8);
  v26 = (_QWORD *)(v25 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v25 & 4) != 0 )
    v26 = (_QWORD *)*v26;
  v27 = 0;
  if ( v8 )
  {
    v36 = v17;
    v37 = v19;
    v40 = v20;
    v28 = sub_B9B140(v26, v38, v8);
    v17 = v36;
    v19 = v37;
    v20 = v40;
    v27 = v28;
  }
  v29 = 0;
  if ( v19 )
  {
    v41 = v20;
    v30 = sub_B9B140(v26, v17, v19);
    v20 = v41;
    v29 = v30;
  }
  v31 = 0;
  if ( v23 )
    v31 = sub_B9B140(v26, v20, v23);
  *a1 = sub_B07EA0(
          (_DWORD)v26,
          v45,
          v31,
          v29,
          v43,
          v48,
          v42,
          v49,
          v47,
          v55,
          v54,
          v53,
          v52,
          v51,
          v50,
          v58,
          v57,
          v56,
          v46,
          v27,
          2,
          1);
  return a1;
}
