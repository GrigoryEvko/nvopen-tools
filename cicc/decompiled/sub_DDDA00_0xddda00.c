// Function: sub_DDDA00
// Address: 0xddda00
//
__int64 __fastcall sub_DDDA00(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, _BYTE *a5)
{
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // eax
  unsigned int v12; // r11d
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  _BYTE *v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 *v23; // r9
  _QWORD *v24; // rax
  _BYTE *v25; // rax
  char v26; // al
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 v29; // r12
  int v30; // eax
  __int64 v31; // rdx
  unsigned int v32; // eax
  __int64 v33; // r13
  unsigned __int64 v34; // rdx
  unsigned int v35; // eax
  __int64 v36; // rdx
  __int64 *v37; // r15
  unsigned int v38; // ecx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rsi
  unsigned int v42; // eax
  __int64 v43; // r12
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // r8
  char v47; // al
  __int64 v48; // [rsp-10h] [rbp-80h]
  __int64 *v49; // [rsp-8h] [rbp-78h]
  __int64 v50; // [rsp+0h] [rbp-70h]
  unsigned __int8 v51; // [rsp+0h] [rbp-70h]
  __int64 v52; // [rsp+8h] [rbp-68h]
  unsigned __int64 v53; // [rsp+8h] [rbp-68h]
  __int64 v54; // [rsp+10h] [rbp-60h]
  __int64 v55; // [rsp+10h] [rbp-60h]
  __int64 v56; // [rsp+10h] [rbp-60h]
  __int64 v57; // [rsp+18h] [rbp-58h]
  _QWORD *v58; // [rsp+20h] [rbp-50h]
  __int64 v59; // [rsp+20h] [rbp-50h]
  __int64 v60; // [rsp+20h] [rbp-50h]
  __int64 *v61; // [rsp+20h] [rbp-50h]
  _QWORD v63[8]; // [rsp+30h] [rbp-40h] BYREF

  if ( a2 )
  {
    v5 = *(_QWORD *)(a1 + 40);
    v6 = a2;
    v9 = **(_QWORD **)(a2 + 32);
    if ( v9 )
    {
      v10 = (unsigned int)(*(_DWORD *)(v9 + 44) + 1);
      v11 = *(_DWORD *)(v9 + 44) + 1;
    }
    else
    {
      v10 = 0;
      v11 = 0;
    }
    if ( v11 < *(_DWORD *)(v5 + 32)
      && *(_QWORD *)(*(_QWORD *)(v5 + 24) + 8 * v10)
      && !(unsigned __int8)sub_DCD020((__int64 *)a1, a3, a4, (__int64)a5) )
    {
      v14 = sub_D47930(a2);
      v12 = 0;
      v52 = v14;
      if ( !v14 )
        return v12;
      v57 = v14 + 48;
      v15 = *(_QWORD *)(v14 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v57 == v15 )
        goto LABEL_64;
      if ( !v15 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
LABEL_64:
        BUG();
      if ( *(_BYTE *)(v15 - 24) != 31
        || (*(_DWORD *)(v15 - 20) & 0x7FFFFFF) != 3
        || (v47 = sub_DDBDC0(a1, a3, a4, a5, *(_QWORD *)(v15 - 120), **(_QWORD **)(a2 + 32) != *(_QWORD *)(v15 - 56), 0),
            v12 = 0,
            !v47) )
      {
        if ( *(_BYTE *)(a1 + 608) )
          return v12;
        *(_BYTE *)(a1 + 608) = 1;
        v16 = sub_DB9E00(a1, a2);
        v17 = v52;
        v18 = sub_D97B30(v16, v52, 0);
        if ( v18 )
          v19 = *(_BYTE **)(v18 + 40);
        else
          v19 = (_BYTE *)sub_D970F0(a1);
        if ( v19 == (_BYTE *)sub_D970F0(a1) )
          goto LABEL_69;
        v54 = sub_D95540((__int64)v19);
        v58 = sub_DA2C50(a1, v54, 1, 0);
        v24 = sub_DA2C50(a1, v54, 0, 0);
        v25 = sub_DC1960(a1, (__int64)v24, (__int64)v58, v6, 3u);
        v17 = a3;
        v26 = sub_DDB9F0((__int64 *)a1, a3, a4, a5, 0x24u, v25, v19, 0);
        v22 = v48;
        v23 = v49;
        if ( !v26 )
        {
LABEL_69:
          v27 = *(_QWORD *)(a1 + 32);
          if ( !*(_BYTE *)(v27 + 192) )
          {
            v60 = *(_QWORD *)(a1 + 32);
            sub_CFDFC0(v60, v17, v20, v21, v22, v23);
            v27 = v60;
          }
          v28 = *(_QWORD *)(v27 + 16);
          v59 = v28 + 32LL * *(unsigned int *)(v27 + 24);
          if ( v59 != v28 )
          {
            v50 = v6;
            v29 = *(_QWORD *)(v27 + 16);
            v55 = a4;
            while ( 1 )
            {
              v33 = *(_QWORD *)(v29 + 16);
              if ( v33 )
              {
                v34 = *(_QWORD *)(v52 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v57 == v34 )
                {
                  v31 = 0;
                }
                else
                {
                  if ( !v34 )
                    BUG();
                  v30 = *(unsigned __int8 *)(v34 - 24);
                  v31 = v34 - 24;
                  if ( (unsigned int)(v30 - 30) >= 0xB )
                    v31 = 0;
                }
                if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 40), *(_QWORD *)(v29 + 16), v31) )
                {
                  v32 = sub_DDBDC0(a1, a3, v55, a5, *(_QWORD *)(v33 - 32LL * (*(_DWORD *)(v33 + 4) & 0x7FFFFFF)), 0, 0);
                  if ( (_BYTE)v32 )
                    break;
                }
              }
              v29 += 32;
              if ( v59 == v29 )
              {
                v6 = v50;
                a4 = v55;
                goto LABEL_35;
              }
            }
            v12 = v32;
            goto LABEL_55;
          }
LABEL_35:
          LOBYTE(v35) = sub_DDC2C0(a1, v52, a3, a4, a5);
          v12 = v35;
          if ( !(_BYTE)v35 )
          {
            v36 = *(_QWORD *)(a1 + 40);
            v37 = 0;
            v38 = *(_DWORD *)(v36 + 32);
            v39 = (unsigned int)(*(_DWORD *)(v52 + 44) + 1);
            if ( (unsigned int)v39 < v38 )
              v37 = *(__int64 **)(*(_QWORD *)(v36 + 24) + 8 * v39);
            v40 = **(_QWORD **)(v6 + 32);
            if ( v40 )
            {
              v41 = (unsigned int)(*(_DWORD *)(v40 + 44) + 1);
              v42 = *(_DWORD *)(v40 + 44) + 1;
            }
            else
            {
              v41 = 0;
              v42 = 0;
            }
            v61 = 0;
            if ( v42 < v38 )
              v61 = *(__int64 **)(*(_QWORD *)(v36 + 24) + 8 * v41);
            if ( v61 == v37 )
              goto LABEL_55;
            v51 = v12;
            while ( 1 )
            {
              v43 = *v37;
              if ( sub_DDC2C0(a1, *v37, a3, a4, a5) )
                break;
              v44 = sub_AA54C0(v43);
              if ( v44 )
              {
                v45 = *(_QWORD *)(v44 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v45 == v44 + 48 )
                  goto LABEL_66;
                if ( !v45 )
                  BUG();
                if ( (unsigned int)*(unsigned __int8 *)(v45 - 24) - 30 > 0xA )
LABEL_66:
                  BUG();
                if ( *(_BYTE *)(v45 - 24) == 31 && (*(_DWORD *)(v45 - 20) & 0x7FFFFFF) == 3 )
                {
                  v46 = *(_QWORD *)(v45 - 120);
                  v53 = *(_QWORD *)(v44 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  v63[0] = v44;
                  v56 = v46;
                  v63[1] = v43;
                  if ( (unsigned __int8)sub_B190C0(v63) )
                  {
                    if ( (unsigned __int8)sub_DDBDC0(a1, a3, a4, a5, v56, *(_QWORD *)(v53 - 56) != v43, 0) )
                      break;
                  }
                }
              }
              v37 = (__int64 *)v37[1];
              if ( v61 == v37 )
              {
                v12 = v51;
                goto LABEL_55;
              }
            }
          }
        }
        v12 = 1;
LABEL_55:
        *(_BYTE *)(a1 + 608) = 0;
        return v12;
      }
    }
  }
  return 1;
}
