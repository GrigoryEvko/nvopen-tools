// Function: sub_33945B0
// Address: 0x33945b0
//
_QWORD *__fastcall sub_33945B0(__int64 a1, unsigned __int8 *a2, int a3)
{
  __int64 *v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // r12
  unsigned __int8 *v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  int v18; // r9d
  unsigned __int64 v19; // rax
  _BOOL4 v20; // esi
  bool v21; // dl
  int v22; // r9d
  __int64 v23; // rax
  unsigned __int16 *v24; // r12
  __int64 v25; // r11
  __int64 v26; // r8
  int v27; // r15d
  __int64 v28; // rsi
  __int64 v29; // r12
  int v30; // edx
  int v31; // r15d
  _QWORD *result; // rax
  __int64 v33; // rax
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // r11
  __int64 v37; // rsi
  unsigned int v38; // edx
  __int64 v39; // rdx
  unsigned __int8 v40; // r9
  int v41; // eax
  unsigned __int8 v42; // r9
  __int128 v43; // [rsp-10h] [rbp-B0h]
  __int64 v44; // [rsp+8h] [rbp-98h]
  __int128 v45; // [rsp+10h] [rbp-90h]
  int v46; // [rsp+20h] [rbp-80h]
  __int64 v47; // [rsp+20h] [rbp-80h]
  unsigned int v48; // [rsp+28h] [rbp-78h]
  int v49; // [rsp+28h] [rbp-78h]
  __int64 v50; // [rsp+28h] [rbp-78h]
  __int64 v51; // [rsp+28h] [rbp-78h]
  __int64 v52; // [rsp+30h] [rbp-70h]
  __int64 v53; // [rsp+30h] [rbp-70h]
  int v54; // [rsp+38h] [rbp-68h]
  __int64 v56; // [rsp+40h] [rbp-60h]
  __int64 v57; // [rsp+48h] [rbp-58h]
  unsigned __int8 *v58; // [rsp+60h] [rbp-40h] BYREF
  int v59; // [rsp+68h] [rbp-38h]

  if ( (a2[7] & 0x40) != 0 )
    v4 = (__int64 *)*((_QWORD *)a2 - 1);
  else
    v4 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  *(_QWORD *)&v45 = sub_338B750(a1, *v4);
  v6 = (unsigned int)v5;
  *((_QWORD *)&v45 + 1) = v5;
  if ( (a2[7] & 0x40) != 0 )
    v7 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v7 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v8 = 16 * v6;
  v56 = sub_338B750(a1, *((_QWORD *)v7 + 4));
  v52 = v56;
  v9 = *(_QWORD *)(a1 + 864);
  v57 = v10;
  v11 = *(_QWORD *)(v9 + 16);
  v48 = v10;
  v12 = sub_2E79000(*(__int64 **)(v9 + 40));
  v13 = sub_2FE6750(
          v11,
          *(unsigned __int16 *)(v8 + *(_QWORD *)(v45 + 48)),
          *(_QWORD *)(v8 + *(_QWORD *)(v45 + 48) + 8),
          v12);
  v15 = v48;
  v16 = v13;
  v17 = v14;
  if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a2 + 1) + 8LL) - 17 > 1 )
  {
    v33 = *(_QWORD *)(v56 + 48) + 16LL * v48;
    if ( *(_WORD *)v33 != (_WORD)v16 || *(_QWORD *)(v33 + 8) != v14 && !*(_WORD *)v33 )
    {
      v34 = *(_DWORD *)(a1 + 848);
      v35 = *(_QWORD *)a1;
      v58 = 0;
      v36 = *(_QWORD *)(a1 + 864);
      v59 = v34;
      if ( v35 )
      {
        if ( &v58 != (unsigned __int8 **)(v35 + 48) )
        {
          v37 = *(_QWORD *)(v35 + 48);
          v58 = (unsigned __int8 *)v37;
          if ( v37 )
          {
            v47 = v16;
            v50 = v17;
            v53 = v36;
            sub_B96E90((__int64)&v58, v37, 1);
            v16 = v47;
            v17 = v50;
            v36 = v53;
          }
        }
      }
      v52 = sub_33FB310(v36, v56, v57, &v58, v16, v17);
      v15 = v38;
      if ( v58 )
      {
        v51 = v38;
        sub_B91220((__int64)&v58, (__int64)v58);
        v15 = v51;
      }
    }
  }
  v18 = 0;
  if ( (unsigned int)(a3 - 190) <= 2 )
  {
    v19 = *a2;
    if ( (unsigned __int8)v19 <= 0x1Cu )
    {
      v20 = 0;
      v22 = 0;
      if ( (_BYTE)v19 != 5 )
        goto LABEL_22;
      v41 = *((unsigned __int16 *)a2 + 1);
      v21 = (*((_WORD *)a2 + 1) & 0xFFFD) == 13 || (*((_WORD *)a2 + 1) & 0xFFF7) == 17;
      if ( v21 )
      {
        v42 = a2[1];
        v21 = (v42 & 4) != 0;
        v20 = (v42 & 2) != 0;
      }
      if ( (unsigned __int16)(v41 - 26) > 1u )
      {
        v22 = 0;
        if ( (unsigned int)(v41 - 19) > 1 )
        {
LABEL_12:
          if ( v21 )
          {
            v18 = v20 | v22 | 2;
            goto LABEL_14;
          }
LABEL_22:
          v18 = v20 | v22;
          goto LABEL_14;
        }
      }
    }
    else
    {
      if ( (unsigned __int8)v19 <= 0x36u && (v39 = 0x40540000000000LL, _bittest64(&v39, v19)) )
      {
        v40 = a2[1];
        v21 = (v40 & 4) != 0;
        v20 = (v40 & 2) != 0;
      }
      else
      {
        v20 = 0;
        v21 = 0;
      }
      if ( (unsigned int)(unsigned __int8)v19 - 48 > 1 )
      {
        v22 = 0;
        if ( (unsigned __int8)(v19 - 55) > 1u )
          goto LABEL_12;
      }
    }
    v22 = 4 * ((a2[1] & 2) != 0);
    goto LABEL_12;
  }
LABEL_14:
  v23 = *(_QWORD *)a1;
  v24 = (unsigned __int16 *)(*(_QWORD *)(v45 + 48) + v8);
  v25 = *(_QWORD *)(a1 + 864);
  v26 = *((_QWORD *)v24 + 1);
  v27 = *v24;
  v59 = *(_DWORD *)(a1 + 848);
  v58 = 0;
  if ( v23 )
  {
    if ( &v58 != (unsigned __int8 **)(v23 + 48) )
    {
      v28 = *(_QWORD *)(v23 + 48);
      v58 = (unsigned __int8 *)v28;
      if ( v28 )
      {
        v44 = v15;
        v54 = v18;
        v46 = v26;
        v49 = v25;
        sub_B96E90((__int64)&v58, v28, 1);
        v15 = v44;
        v18 = v54;
        LODWORD(v26) = v46;
        LODWORD(v25) = v49;
      }
    }
  }
  *((_QWORD *)&v43 + 1) = v15 | v57 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v43 = v52;
  v29 = sub_3405C90(v25, a3, (unsigned int)&v58, v27, v26, v18, v45, v43);
  v31 = v30;
  if ( v58 )
    sub_B91220((__int64)&v58, (__int64)v58);
  v58 = a2;
  result = sub_337DC20(a1 + 8, (__int64 *)&v58);
  *result = v29;
  *((_DWORD *)result + 2) = v31;
  return result;
}
