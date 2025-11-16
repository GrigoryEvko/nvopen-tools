// Function: sub_328C2D0
// Address: 0x328c2d0
//
__int64 __fastcall sub_328C2D0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // r14d
  __int128 v5; // xmm1
  unsigned __int16 *v6; // rax
  int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r11
  __int64 v11; // rsi
  __int64 result; // rax
  int v13; // r9d
  __int64 v14; // rsi
  __int64 v15; // r12
  int v16; // esi
  unsigned int v17; // r15d
  __int64 v18; // rax
  void *v19; // rdx
  char v20; // al
  _QWORD **v21; // rsi
  char v22; // al
  __int64 v23; // rcx
  void *v24; // rdx
  _QWORD **v25; // rax
  __int64 v26; // rsi
  unsigned int v27; // esi
  __int64 v28; // r12
  char v29; // si
  __int64 *v30; // rdx
  void *v31; // [rsp+0h] [rbp-E0h]
  void *v32; // [rsp+0h] [rbp-E0h]
  int v33; // [rsp+10h] [rbp-D0h]
  __int64 v34; // [rsp+10h] [rbp-D0h]
  __int64 v35; // [rsp+10h] [rbp-D0h]
  __int64 v36; // [rsp+10h] [rbp-D0h]
  __int64 v37; // [rsp+18h] [rbp-C8h]
  __int64 v38; // [rsp+20h] [rbp-C0h]
  int v39; // [rsp+2Ch] [rbp-B4h]
  __int64 v40; // [rsp+38h] [rbp-A8h]
  __int128 v41; // [rsp+40h] [rbp-A0h]
  unsigned int v42; // [rsp+40h] [rbp-A0h]
  __int64 v43; // [rsp+50h] [rbp-90h]
  __int64 v44; // [rsp+50h] [rbp-90h]
  __int64 v45; // [rsp+50h] [rbp-90h]
  _QWORD **v46; // [rsp+50h] [rbp-90h]
  __int64 v47; // [rsp+60h] [rbp-80h] BYREF
  int v48; // [rsp+68h] [rbp-78h]
  __int64 v49; // [rsp+70h] [rbp-70h] BYREF
  int v50; // [rsp+78h] [rbp-68h]
  __int64 v51; // [rsp+80h] [rbp-60h]
  _OWORD v52[5]; // [rsp+90h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_DWORD *)(a2 + 24);
  v5 = (__int128)_mm_loadu_si128((const __m128i *)(v3 + 40));
  v38 = *(_QWORD *)v3;
  v37 = *(_QWORD *)(v3 + 40);
  v6 = *(unsigned __int16 **)(a2 + 48);
  v41 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v7 = *v6;
  v40 = *((_QWORD *)v6 + 1);
  v39 = *(_DWORD *)(a2 + 28);
  v8 = *a1;
  v50 = v39;
  v9 = *(_QWORD *)(v8 + 1024);
  v49 = v8;
  v51 = v9;
  *(_QWORD *)(v8 + 1024) = &v49;
  v10 = *a1;
  v52[0] = v41;
  v52[1] = v5;
  v11 = *(_QWORD *)(a2 + 80);
  v47 = v11;
  if ( v11 )
  {
    v33 = v10;
    sub_B96E90((__int64)&v47, v11, 1);
    LODWORD(v10) = v33;
  }
  v48 = *(_DWORD *)(a2 + 72);
  result = sub_3402EA0(v10, v4, (unsigned int)&v47, v7, v40, 0, (__int64)v52, 2);
  if ( v47 )
  {
    v34 = result;
    sub_B91220((__int64)&v47, v47);
    result = v34;
  }
  if ( !result )
  {
    if ( (unsigned __int8)sub_33E2470(*a1, v41, *((_QWORD *)&v41 + 1))
      && !(unsigned __int8)sub_33E2470(*a1, v5, *((_QWORD *)&v5 + 1)) )
    {
      v14 = *(_QWORD *)(a2 + 80);
      v15 = *a1;
      *(_QWORD *)&v52[0] = v14;
      if ( v14 )
        sub_B96E90((__int64)v52, v14, 1);
      v16 = *(_DWORD *)(a2 + 24);
      DWORD2(v52[0]) = *(_DWORD *)(a2 + 72);
      result = sub_3406EB0(v15, v16, (unsigned int)v52, v7, v40, v13, v5, v41);
      if ( *(_QWORD *)&v52[0] )
      {
        v43 = result;
        sub_B91220((__int64)v52, *(__int64 *)&v52[0]);
        result = v43;
      }
      goto LABEL_6;
    }
    v17 = v4 - 283;
    v42 = (v4 - 279) & 0xFFFFFFFB;
    v18 = sub_33E1790(v5, *((_QWORD *)&v5 + 1), 0);
    if ( !v18 )
      goto LABEL_34;
    v44 = *(_QWORD *)(v18 + 96);
    v19 = sub_C33340();
    if ( *(void **)(v44 + 24) == v19 )
    {
      v25 = *(_QWORD ***)(v44 + 32);
      v29 = *((_BYTE *)v25 + 20) & 7;
      if ( v29 != 1 )
      {
        if ( !v29 )
          goto LABEL_22;
        if ( (v39 & 0x40) == 0 )
          goto LABEL_34;
        v32 = v19;
        v36 = v44;
        v46 = (_QWORD **)(v44 + 24);
        v22 = sub_C40510(v46);
        v21 = v46;
        v23 = v36;
        v24 = v32;
        goto LABEL_19;
      }
    }
    else
    {
      v20 = *(_BYTE *)(v44 + 44) & 7;
      if ( v20 != 1 )
      {
        v21 = (_QWORD **)(v44 + 24);
        if ( !v20 )
        {
LABEL_21:
          v25 = v21;
LABEL_22:
          if ( (v42 == 0) != ((*((_BYTE *)v25 + 20) & 8) != 0) )
          {
            if ( v17 > 1 && (v39 & 0x20) == 0 )
            {
              v26 = *(_QWORD *)(a2 + 80);
              *(_QWORD *)&v52[0] = v26;
              if ( !v26 )
              {
                DWORD2(v52[0]) = *(_DWORD *)(a2 + 72);
                goto LABEL_28;
              }
              goto LABEL_26;
            }
            v30 = *(__int64 **)(a2 + 40);
LABEL_45:
            result = *v30;
            goto LABEL_6;
          }
          if ( v17 <= 1 && (v39 & 0x20) == 0 )
          {
            v26 = *(_QWORD *)(a2 + 80);
            *(_QWORD *)&v52[0] = v26;
            if ( !v26 )
            {
              DWORD2(v52[0]) = *(_DWORD *)(a2 + 72);
              goto LABEL_43;
            }
            goto LABEL_26;
          }
          v30 = *(__int64 **)(a2 + 40);
LABEL_48:
          result = v30[5];
          goto LABEL_6;
        }
        if ( (v39 & 0x40) == 0 )
        {
LABEL_34:
          v26 = *(_QWORD *)(a2 + 80);
          *(_QWORD *)&v52[0] = v26;
          if ( !v26 )
          {
LABEL_27:
            DWORD2(v52[0]) = *(_DWORD *)(a2 + 72);
            if ( v17 > 1 )
            {
LABEL_28:
              v27 = (v42 == 0) + 378;
LABEL_29:
              v28 = sub_328C120(a1, v27, v4, (int)v52, v7, v40, v38, v37, v39);
              if ( *(_QWORD *)&v52[0] )
                sub_B91220((__int64)v52, *(__int64 *)&v52[0]);
              result = 0;
              if ( v28 )
                result = v28;
              goto LABEL_6;
            }
LABEL_43:
            v27 = (v42 == 0) + 380;
            goto LABEL_29;
          }
LABEL_26:
          sub_B96E90((__int64)v52, v26, 1);
          goto LABEL_27;
        }
        v31 = v19;
        v35 = v44;
        v45 = v44 + 24;
        v22 = sub_C33B00(v45);
        v21 = (_QWORD **)v45;
        v23 = v35;
        v24 = v31;
LABEL_19:
        if ( v22 )
        {
          if ( v24 == *(void **)(v23 + 24) )
          {
            v25 = *(_QWORD ***)(v23 + 32);
            goto LABEL_22;
          }
          goto LABEL_21;
        }
        goto LABEL_34;
      }
    }
    v30 = *(__int64 **)(a2 + 40);
    if ( v17 > 1 )
      goto LABEL_45;
    goto LABEL_48;
  }
LABEL_6:
  *(_QWORD *)(v49 + 1024) = v51;
  return result;
}
