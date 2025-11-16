// Function: sub_321C870
// Address: 0x321c870
//
__int64 __fastcall sub_321C870(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  unsigned int v11; // r12d
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  _BYTE *v21; // rax
  _BYTE *v22; // rsi
  __int64 v23; // rdx
  signed __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int8 v27; // dl
  __int64 *v28; // rax
  __int64 v29; // rax
  unsigned __int8 v30; // si
  _QWORD *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // [rsp+0h] [rbp-70h]
  unsigned __int64 v36; // [rsp+10h] [rbp-60h]
  unsigned __int64 v37; // [rsp+10h] [rbp-60h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+18h] [rbp-58h]
  __int64 v41; // [rsp+28h] [rbp-48h]
  __int64 v42; // [rsp+30h] [rbp-40h] BYREF
  __int64 v43[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = *(_QWORD *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 56);
  v41 = v6;
  v42 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v42, v7, 1);
  v8 = sub_B10CD0((__int64)&v42);
  v9 = sub_35051D0(a1, v8);
  v10 = v9;
  if ( !v9 || !*(_DWORD *)(v9 + 88) )
    goto LABEL_5;
  v39 = **(_QWORD **)(v9 + 80);
  if ( !(unsigned __int8)sub_372AA30(a4, a2) )
  {
    if ( *(_QWORD *)(v39 + 24) != v41 )
    {
LABEL_5:
      v11 = 0;
      goto LABEL_6;
    }
    v13 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v13 )
      BUG();
    v14 = *(_QWORD *)v13;
    if ( (*(_QWORD *)v13 & 4) == 0 && (*(_BYTE *)(v13 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        v34 = v14 & 0xFFFFFFFFFFFFFFF8LL;
        v13 = v34;
        if ( (*(_BYTE *)(v34 + 44) & 4) == 0 )
          break;
        v14 = *(_QWORD *)v34;
      }
    }
    if ( v41 + 48 != v13 )
    {
      while ( (*(_BYTE *)(v13 + 44) & 1) == 0 )
      {
        v15 = *(_QWORD *)(v13 + 56);
        v43[0] = v15;
        if ( v15 )
        {
          v36 = v13;
          sub_B96E90((__int64)v43, v15, 1);
          v16 = v43[0];
          v13 = v36;
          if ( v43[0] )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(v36 + 16) + 24LL) & 0x10) != 0 )
              goto LABEL_18;
            v26 = sub_B10CD0((__int64)&v42);
            v27 = *(_BYTE *)(v26 - 16);
            if ( (v27 & 2) != 0 )
              v28 = *(__int64 **)(v26 - 32);
            else
              v28 = (__int64 *)(v26 - 16 - 8LL * ((v27 >> 2) & 0xF));
            v35 = v36;
            v38 = *v28;
            v29 = sub_B10CD0((__int64)v43);
            v30 = *(_BYTE *)(v29 - 16);
            if ( (v30 & 2) != 0 )
              v31 = *(_QWORD **)(v29 - 32);
            else
              v31 = (_QWORD *)(v29 - 16 - 8LL * ((v30 >> 2) & 0xF));
            if ( *v31 == v38
              || (v32 = sub_B10CD0((__int64)v43), v33 = sub_35051D0(a1, v32), v10 == v33)
              || !v33
              || (v13 = v35, *(_DWORD *)(v10 + 176) < *(_DWORD *)(v33 + 176))
              && *(_DWORD *)(v10 + 180) > *(_DWORD *)(v33 + 180) )
            {
              if ( v43[0] )
                sub_B91220((__int64)v43, v43[0]);
              goto LABEL_5;
            }
            v16 = v43[0];
            if ( v43[0] )
            {
LABEL_18:
              v37 = v13;
              sub_B91220((__int64)v43, v16);
              v13 = v37;
            }
          }
        }
        v17 = (_QWORD *)(*(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL);
        v18 = v17;
        if ( !v17 )
          BUG();
        v13 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL;
        v19 = *v17;
        if ( (v19 & 4) == 0 && (*((_BYTE *)v18 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            v20 = v19 & 0xFFFFFFFFFFFFFFF8LL;
            v13 = v20;
            if ( (*(_BYTE *)(v20 + 44) & 4) == 0 )
              break;
            v19 = *(_QWORD *)v20;
          }
        }
        if ( v13 == v41 + 48 )
          break;
      }
    }
  }
  if ( !a3 )
    goto LABEL_39;
  if ( *(_DWORD *)(v41 + 72) )
  {
LABEL_37:
    v11 = sub_372AA30(a4, a3) ^ 1;
    goto LABEL_6;
  }
  v21 = *(_BYTE **)(a2 + 32);
  v22 = v21 + 40;
  if ( *(_WORD *)(a2 + 68) == 14 )
    goto LABEL_70;
  v23 = 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  v22 = &v21[v23];
  v21 += 80;
  v24 = 0xCCCCCCCCCCCCCCCDLL * ((v23 - 80) >> 3);
  v25 = v24 >> 2;
  if ( v24 >> 2 > 0 )
  {
    while ( *v21 == 1 )
    {
      if ( v21[40] != 1 )
      {
        v21 += 40;
        break;
      }
      if ( v21[80] != 1 )
      {
        v21 += 80;
        break;
      }
      if ( v21[120] != 1 )
      {
        v21 += 120;
        break;
      }
      v21 += 160;
      if ( !--v25 )
      {
        v24 = 0xCCCCCCCCCCCCCCCDLL * ((v22 - v21) >> 3);
        goto LABEL_64;
      }
    }
LABEL_36:
    v11 = 1;
    if ( v21 == v22 )
      goto LABEL_6;
    goto LABEL_37;
  }
LABEL_64:
  if ( v24 != 2 )
  {
    if ( v24 == 3 )
    {
      if ( *v21 != 1 )
        goto LABEL_36;
      v21 += 40;
      goto LABEL_68;
    }
    if ( v24 == 1 )
      goto LABEL_70;
LABEL_39:
    v11 = 1;
    goto LABEL_6;
  }
LABEL_68:
  if ( *v21 != 1 )
    goto LABEL_36;
  v21 += 40;
LABEL_70:
  v11 = 1;
  if ( *v21 != 1 )
    goto LABEL_36;
LABEL_6:
  if ( v42 )
    sub_B91220((__int64)&v42, v42);
  return v11;
}
