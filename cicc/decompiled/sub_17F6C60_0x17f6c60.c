// Function: sub_17F6C60
// Address: 0x17f6c60
//
void __fastcall sub_17F6C60(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r13
  __int64 *i; // r12
  unsigned __int8 *v5; // rsi
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int8 *v8; // rsi
  __int64 v9; // rax
  _BYTE *v10; // r9
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rbx
  char v14; // cl
  __int64 v15; // r15
  __int64 **v16; // rax
  __int64 v17; // r9
  __int64 **v18; // r11
  __int64 ***v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rsi
  _BYTE *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdx
  unsigned __int8 *v27; // rsi
  _QWORD *v28; // rdi
  __int64 v29; // rax
  __int64 **v30; // r11
  __int64 v31; // r9
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int8 *v36; // rsi
  __int64 **v37; // [rsp+10h] [rbp-140h]
  __int64 *v38; // [rsp+18h] [rbp-138h]
  __int64 **v39; // [rsp+18h] [rbp-138h]
  __int64 **v40; // [rsp+18h] [rbp-138h]
  _BYTE *v41; // [rsp+20h] [rbp-130h]
  __int64 **v42; // [rsp+20h] [rbp-130h]
  __int64 *v43; // [rsp+20h] [rbp-130h]
  __int64 **v44; // [rsp+20h] [rbp-130h]
  __int64 v45; // [rsp+20h] [rbp-130h]
  __int64 v46; // [rsp+20h] [rbp-130h]
  __int64 v47; // [rsp+20h] [rbp-130h]
  unsigned __int8 *v49; // [rsp+38h] [rbp-118h] BYREF
  _QWORD v50[2]; // [rsp+40h] [rbp-110h] BYREF
  __int64 v51[2]; // [rsp+50h] [rbp-100h] BYREF
  __int16 v52; // [rsp+60h] [rbp-F0h]
  __int64 v53; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v54; // [rsp+80h] [rbp-D0h]
  char v55[16]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v56; // [rsp+A0h] [rbp-B0h]
  unsigned __int8 *v57[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v58; // [rsp+C0h] [rbp-90h]
  unsigned __int8 *v59; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v60; // [rsp+D8h] [rbp-78h]
  __int64 *v61; // [rsp+E0h] [rbp-70h]
  __int64 v62; // [rsp+E8h] [rbp-68h]
  __int64 v63; // [rsp+F0h] [rbp-60h]
  int v64; // [rsp+F8h] [rbp-58h]
  __int64 v65; // [rsp+100h] [rbp-50h]
  __int64 v66; // [rsp+108h] [rbp-48h]

  v3 = &a2[a3];
  if ( a2 != v3 )
  {
    for ( i = a2; v3 != i; ++i )
    {
      v6 = *i;
      if ( *(_BYTE *)(*i + 16) != 75 )
        continue;
      v7 = sub_16498A0(*i);
      v65 = 0;
      v66 = 0;
      v8 = *(unsigned __int8 **)(v6 + 48);
      v62 = v7;
      v64 = 0;
      v9 = *(_QWORD *)(v6 + 40);
      v59 = 0;
      v60 = v9;
      v63 = 0;
      v61 = (__int64 *)(v6 + 24);
      v57[0] = v8;
      if ( v8 )
      {
        sub_1623A60((__int64)v57, (__int64)v8, 2);
        if ( v59 )
          sub_161E7C0((__int64)&v59, (__int64)v59);
        v59 = v57[0];
        if ( v57[0] )
          sub_1623210((__int64)v57, v57[0], (__int64)&v59);
      }
      v10 = *(_BYTE **)(v6 - 48);
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 11 )
      {
        v11 = ((*(_DWORD *)(*(_QWORD *)v10 + 8LL) >> 8) + 7) & 0x1FFFFF8;
        switch ( v11 )
        {
          case 8LL:
            v12 = 0;
            break;
          case 16LL:
            v12 = 1;
            break;
          case 32LL:
            v12 = 2;
            break;
          default:
            v12 = 3;
            if ( v11 != 64 )
              goto LABEL_3;
            break;
        }
        v13 = *(_QWORD *)(v6 - 24);
        v14 = *(_BYTE *)(v13 + 16);
        if ( v10[16] != 13 || v14 != 13 )
        {
          v15 = *(_QWORD *)(a1 + 8 * v12 + 184);
          if ( v10[16] == 13 || v14 == 13 )
          {
            v15 = *(_QWORD *)(a1 + 8 * v12 + 216);
            if ( v14 == 13 )
            {
              v22 = v10;
              v10 = (_BYTE *)v13;
              v13 = (__int64)v22;
            }
          }
          v41 = v10;
          v16 = (__int64 **)sub_1644C60(*(_QWORD **)(a1 + 432), v11);
          v54 = 257;
          v17 = (__int64)v41;
          v18 = v16;
          v56 = 257;
          if ( v16 != *(__int64 ***)v41 )
          {
            if ( v41[16] > 0x10u )
            {
              v58 = 257;
              v28 = v41;
              v44 = v16;
              v29 = sub_15FE0A0(v28, (__int64)v16, 1, (__int64)v57, 0);
              v30 = v44;
              v31 = v29;
              if ( v60 )
              {
                v37 = v44;
                v45 = v29;
                v38 = v61;
                sub_157E9D0(v60 + 40, v29);
                v31 = v45;
                v30 = v37;
                v32 = *v38;
                v33 = *(_QWORD *)(v45 + 24);
                *(_QWORD *)(v45 + 32) = v38;
                v32 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v45 + 24) = v32 | v33 & 7;
                *(_QWORD *)(v32 + 8) = v45 + 24;
                *v38 = *v38 & 7 | (v45 + 24);
              }
              v39 = v30;
              v46 = v31;
              sub_164B780(v31, &v53);
              v17 = v46;
              v18 = v39;
              if ( v59 )
              {
                v51[0] = (__int64)v59;
                sub_1623A60((__int64)v51, (__int64)v59, 2);
                v17 = v46;
                v18 = v39;
                v34 = *(_QWORD *)(v46 + 48);
                v35 = v46 + 48;
                if ( v34 )
                {
                  sub_161E7C0(v46 + 48, v34);
                  v18 = v39;
                  v17 = v46;
                  v35 = v46 + 48;
                }
                v36 = (unsigned __int8 *)v51[0];
                *(_QWORD *)(v17 + 48) = v51[0];
                if ( v36 )
                {
                  v40 = v18;
                  v47 = v17;
                  sub_1623210((__int64)v51, v36, v35);
                  v18 = v40;
                  v17 = v47;
                }
              }
            }
            else
            {
              v19 = (__int64 ***)v41;
              v42 = v16;
              v20 = sub_15A4750(v19, v16, 1);
              v18 = v42;
              v17 = v20;
            }
          }
          v50[0] = v17;
          v52 = 257;
          if ( v18 != *(__int64 ***)v13 )
          {
            if ( *(_BYTE *)(v13 + 16) > 0x10u )
            {
              v58 = 257;
              v13 = sub_15FE0A0((_QWORD *)v13, (__int64)v18, 1, (__int64)v57, 0);
              if ( v60 )
              {
                v43 = v61;
                sub_157E9D0(v60 + 40, v13);
                v23 = *v43;
                v24 = *(_QWORD *)(v13 + 24) & 7LL;
                *(_QWORD *)(v13 + 32) = v43;
                v23 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v13 + 24) = v23 | v24;
                *(_QWORD *)(v23 + 8) = v13 + 24;
                *v43 = *v43 & 7 | (v13 + 24);
              }
              sub_164B780(v13, v51);
              if ( v59 )
              {
                v49 = v59;
                sub_1623A60((__int64)&v49, (__int64)v59, 2);
                v25 = *(_QWORD *)(v13 + 48);
                v26 = v13 + 48;
                if ( v25 )
                {
                  sub_161E7C0(v13 + 48, v25);
                  v26 = v13 + 48;
                }
                v27 = v49;
                *(_QWORD *)(v13 + 48) = v49;
                if ( v27 )
                  sub_1623210((__int64)&v49, v27, v26);
              }
            }
            else
            {
              v13 = sub_15A4750((__int64 ***)v13, v18, 1);
            }
          }
          v21 = *(_QWORD *)(v15 + 24);
          v50[1] = v13;
          sub_1285290((__int64 *)&v59, v21, v15, (int)v50, 2, (__int64)v55, 0);
          v5 = v59;
          if ( !v59 )
            continue;
LABEL_4:
          sub_161E7C0((__int64)&v59, (__int64)v5);
          continue;
        }
      }
LABEL_3:
      v5 = v59;
      if ( v59 )
        goto LABEL_4;
    }
  }
}
