// Function: sub_373FFB0
// Address: 0x373ffb0
//
unsigned __int64 __fastcall sub_373FFB0(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  unsigned __int64 v7; // r15
  unsigned __int8 v9; // al
  __int64 v10; // r13
  __int64 v11; // rdx
  unsigned __int8 *v12; // r8
  __int64 v13; // rax
  unsigned __int8 *v14; // r8
  __int16 v15; // ax
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  __int64 v18; // r8
  unsigned __int8 v19; // al
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int8 v22; // al
  __int64 v23; // r11
  unsigned __int8 v24; // al
  __int64 v25; // rdx
  int v26; // r8d
  unsigned __int8 v27; // al
  __int64 v28; // r13
  __int64 v29; // rdx
  unsigned __int8 *v30; // rax
  __int16 v31; // ax
  __int64 v32; // rdx
  __int64 v33; // r8
  __int16 v34; // r11
  unsigned __int64 v35; // r9
  __int64 v36; // rcx
  __int64 v37; // r9
  __int64 v38; // rdi
  __int64 v39; // rdx
  unsigned __int8 v40; // al
  __int64 v41; // rdx
  const void *v42; // rcx
  size_t v43; // rdx
  size_t v44; // r8
  unsigned __int8 v45; // al
  __int64 v46; // rdx
  _BYTE *v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // [rsp+8h] [rbp-68h]
  __int64 v51; // [rsp+10h] [rbp-60h]
  unsigned __int8 *v52; // [rsp+10h] [rbp-60h]
  unsigned __int8 *v53; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v54; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v55; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v56; // [rsp+18h] [rbp-58h]
  __int64 v57; // [rsp+18h] [rbp-58h]
  __int16 v58; // [rsp+18h] [rbp-58h]
  __int64 v59; // [rsp+20h] [rbp-50h]

  v7 = (unsigned __int64)sub_3247C80((__int64)a1, (unsigned __int8 *)a2);
  if ( v7 )
    return v7;
  v9 = *(_BYTE *)(a2 - 16);
  v10 = a2 - 16;
  if ( (v9 & 2) != 0 )
    v11 = *(_QWORD *)(a2 - 32);
  else
    v11 = v10 - 8LL * ((v9 >> 2) & 0xF);
  v12 = *(unsigned __int8 **)v11;
  v59 = *(_QWORD *)(v11 + 24);
  if ( *(_QWORD *)v11 && *v12 == 33 )
  {
    v53 = *(unsigned __int8 **)v11;
    v13 = sub_373FD70(a1, (__int64)v12, a3, a4);
    v14 = v53;
    v7 = v13;
    if ( v13 )
    {
LABEL_8:
      v15 = sub_AF18C0(a2);
      v7 = sub_324C6D0(a1, v15, v7, (unsigned __int8 *)a2);
      goto LABEL_9;
    }
  }
  else
  {
    v55 = *(unsigned __int8 **)v11;
    v30 = sub_373FC60(a1, v12);
    v14 = v55;
    v7 = (unsigned __int64)v30;
    if ( v30 )
      goto LABEL_8;
  }
  if ( *v14 == 19 )
  {
    v56 = v14;
    if ( (unsigned __int16)sub_AF18C0(a2) == 52 )
    {
      v31 = sub_AF18C0(a2);
      v32 = a1[11];
      v33 = (__int64)v56;
      a1[21] += 48;
      v34 = v31;
      v35 = (v32 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( a1[12] >= v35 + 48 && v32 )
      {
        a1[11] = v35 + 48;
        v7 = (v32 + 15) & 0xFFFFFFFFFFFFFFF0LL;
        if ( !v35 )
        {
LABEL_34:
          v57 = v33;
          sub_324C3F0((__int64)a1, (unsigned __int8 *)a2, v7);
          sub_3251400((__int64)a1, v57, v7, v36, v57, v37);
          goto LABEL_9;
        }
      }
      else
      {
        v52 = v56;
        v58 = v31;
        v49 = sub_9D1E70((__int64)(a1 + 11), 48, 48, 4);
        v33 = (__int64)v52;
        v34 = v58;
        v7 = v49;
      }
      *(_QWORD *)(v7 + 8) = 0;
      *(_QWORD *)(v7 + 16) = 0;
      *(_QWORD *)v7 = v7 | 4;
      *(_DWORD *)(v7 + 24) = -1;
      *(_WORD *)(v7 + 28) = v34;
      *(_BYTE *)(v7 + 30) = 0;
      *(_QWORD *)(v7 + 32) = 0;
      *(_QWORD *)(v7 + 40) = 0;
      goto LABEL_34;
    }
  }
LABEL_9:
  v16 = *(_BYTE *)(a2 - 16);
  if ( (v16 & 2) != 0 )
  {
    v17 = *(_QWORD *)(a2 - 32);
    v18 = *(_QWORD *)(v17 + 48);
    if ( v18 )
      goto LABEL_11;
  }
  else
  {
    v17 = v10 - 8LL * ((v16 >> 2) & 0xF);
    v18 = *(_QWORD *)(v17 + 48);
    if ( v18 )
    {
LABEL_11:
      v19 = *(_BYTE *)(v18 - 16);
      if ( (v19 & 2) != 0 )
        v20 = *(_QWORD *)(v18 - 32);
      else
        v20 = v18 - 16 - 8LL * ((v19 >> 2) & 0xF);
      v50 = v18 - 16;
      v51 = v18;
      v54 = *(unsigned __int8 **)(v20 + 8);
      v21 = sub_324F7C0(a1, v18);
      sub_32494F0(a1, v7, 71, v21);
      v22 = *(_BYTE *)(v51 - 16);
      if ( (v22 & 2) != 0 )
        v23 = *(_QWORD *)(v51 - 32);
      else
        v23 = v50 - 8LL * ((v22 >> 2) & 0xF);
      if ( v59 != *(_QWORD *)(v23 + 24) )
        sub_32495E0(a1, v7, v59, 73);
      if ( !*(_BYTE *)(a2 + 21) )
        goto LABEL_18;
LABEL_49:
      v45 = *(_BYTE *)(a2 - 16);
      if ( (v45 & 2) != 0 )
        v46 = *(_QWORD *)(a2 - 32);
      else
        v46 = v10 - 8LL * ((v45 >> 2) & 0xF);
      v47 = *(_BYTE **)(v46 + 8);
      if ( v47 )
        v47 = (_BYTE *)sub_B91420(*(_QWORD *)(v46 + 8));
      else
        v48 = 0;
      sub_3736650((__int64)a1, v47, v48, v7, v54);
      v24 = *(_BYTE *)(a2 - 16);
      if ( (v24 & 2) != 0 )
        goto LABEL_19;
LABEL_54:
      v25 = v10 - 8LL * ((v24 >> 2) & 0xF);
      goto LABEL_20;
    }
  }
  v38 = *(_QWORD *)(v17 + 32);
  v54 = *(unsigned __int8 **)v17;
  if ( v38 )
  {
    sub_B91420(v38);
    if ( v39 )
    {
      v40 = *(_BYTE *)(a2 - 16);
      if ( (v40 & 2) != 0 )
        v41 = *(_QWORD *)(a2 - 32);
      else
        v41 = v10 - 8LL * ((v40 >> 2) & 0xF);
      v42 = *(const void **)(v41 + 32);
      if ( v42 )
      {
        v42 = (const void *)sub_B91420(*(_QWORD *)(v41 + 32));
        v44 = v43;
      }
      else
      {
        v44 = 0;
      }
      sub_324AD70(a1, v7, 3, v42, v44);
    }
  }
  if ( v59 )
    sub_32495E0(a1, v7, v59, 73);
  if ( !*(_BYTE *)(a2 + 20) )
    sub_3249FA0(a1, v7, 63);
  sub_3249D50(a1, v7, a2);
  if ( *(_BYTE *)(a2 + 21) )
    goto LABEL_49;
LABEL_18:
  sub_3249FA0(a1, v7, 60);
  v24 = *(_BYTE *)(a2 - 16);
  if ( (v24 & 2) == 0 )
    goto LABEL_54;
LABEL_19:
  v25 = *(_QWORD *)(a2 - 32);
LABEL_20:
  sub_324CC60(a1, v7, *(_QWORD *)(v25 + 64));
  v26 = *(_DWORD *)(a2 + 4) >> 3;
  if ( v26 )
    sub_3249A20(a1, (unsigned __int64 **)(v7 + 8), 136, 65551, v26 & 0x1FFFFFFF);
  v27 = *(_BYTE *)(a2 - 16);
  if ( (v27 & 2) != 0 )
    v28 = *(_QWORD *)(a2 - 32);
  else
    v28 = v10 - 8LL * ((v27 >> 2) & 0xF);
  v29 = *(_QWORD *)(v28 + 56);
  if ( v29 )
    sub_324D230(a1, v7, v29);
  sub_3739A60(a1, v7, a2, a3, a4);
  return v7;
}
