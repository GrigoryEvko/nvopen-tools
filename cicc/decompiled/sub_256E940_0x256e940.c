// Function: sub_256E940
// Address: 0x256e940
//
__int64 __fastcall sub_256E940(__int64 a1, _QWORD *a2)
{
  char *v3; // r14
  __int64 v4; // r13
  __int64 v6; // rax
  char v7; // r15
  char v8; // al
  __int64 v9; // r13
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int8 *v13; // rax
  unsigned __int8 *v14; // r10
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r13
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int8 *v22; // rax
  unsigned __int8 *v23; // r10
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // r13
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  unsigned __int8 *v33; // rax
  unsigned __int8 *v34; // r11
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // [rsp-88h] [rbp-88h]
  __int64 v43; // [rsp-88h] [rbp-88h]
  __int64 v44; // [rsp-88h] [rbp-88h]
  unsigned __int8 *v45; // [rsp-80h] [rbp-80h]
  unsigned __int8 *v46; // [rsp-80h] [rbp-80h]
  unsigned __int8 *v47; // [rsp-80h] [rbp-80h]
  char v48; // [rsp-78h] [rbp-78h]
  unsigned __int8 *v49; // [rsp-78h] [rbp-78h]
  unsigned __int8 *v50; // [rsp-78h] [rbp-78h]
  __int64 v51; // [rsp-78h] [rbp-78h]
  __int64 v52; // [rsp-78h] [rbp-78h]
  __int64 v53; // [rsp-70h] [rbp-70h]
  char v54; // [rsp-70h] [rbp-70h]
  unsigned __int8 *v55; // [rsp-70h] [rbp-70h]
  unsigned __int8 *v56; // [rsp-70h] [rbp-70h]
  char v57; // [rsp-70h] [rbp-70h]
  unsigned __int8 *v58; // [rsp-70h] [rbp-70h]
  unsigned __int8 *v59; // [rsp-70h] [rbp-70h]
  _BYTE v60[32]; // [rsp-68h] [rbp-68h] BYREF
  __int16 v61; // [rsp-48h] [rbp-48h]

  if ( *a2 != **(_QWORD **)a1 )
    return 1;
  v3 = (char *)a2[3];
  if ( (unsigned __int8)*v3 > 0x1Cu )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v6 = sub_B43CB0(a2[3]);
    v7 = sub_253A110(*(_QWORD *)(v4 + 200), v6);
    if ( v7 )
    {
      v8 = *v3;
      if ( *v3 == 61 )
      {
        v48 = **(_BYTE **)(a1 + 40);
        v9 = **(_QWORD **)(a1 + 32);
        v45 = **(unsigned __int8 ***)(a1 + 24);
        v53 = *(_QWORD *)(a1 + 8);
        if ( !(unsigned int)sub_BD2910((__int64)a2) )
        {
          if ( (v3[2] & 1) == 0
            || (v42 = *(_QWORD *)(v53 + 208),
                v27 = sub_B43CB0((__int64)v3),
                (v28 = sub_255ED30(*(_QWORD *)(v42 + 240), v27, 0)) != 0)
            && (unsigned __int8)sub_DFA750(v28) )
          {
            if ( v48 )
            {
              sub_256E5A0(v53, (__int64)a2, v45, v10, v11, v12);
              v7 = v48;
            }
            else
            {
              v61 = 257;
              v13 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
              v14 = v13;
              if ( v13 )
              {
                v49 = v13;
                sub_B51C90((__int64)v13, (__int64)v45, v9, (__int64)v60, 0, 0);
                v14 = v49;
              }
              v50 = v14;
              sub_B44220(v14, (__int64)(v3 + 24), 0);
              sub_256E5A0(v53, (__int64)a2, v50, v15, v16, v17);
            }
            goto LABEL_15;
          }
        }
        goto LABEL_14;
      }
      switch ( v8 )
      {
        case '>':
          v18 = *(_QWORD *)(a1 + 8);
          v54 = **(_BYTE **)(a1 + 40);
          v51 = **(_QWORD **)(a1 + 32);
          v46 = **(unsigned __int8 ***)(a1 + 24);
          if ( (unsigned int)sub_BD2910((__int64)a2) != 1 )
            goto LABEL_14;
          break;
        case 'B':
          v18 = *(_QWORD *)(a1 + 8);
          v54 = **(_BYTE **)(a1 + 40);
          v51 = **(_QWORD **)(a1 + 32);
          v46 = **(unsigned __int8 ***)(a1 + 24);
          if ( (unsigned int)sub_BD2910((__int64)a2) )
          {
LABEL_14:
            v7 = 0;
LABEL_15:
            **(_BYTE **)(a1 + 16) |= v7;
            return 1;
          }
          break;
        case 'A':
          v57 = **(_BYTE **)(a1 + 40);
          v29 = **(_QWORD **)(a1 + 32);
          v47 = **(unsigned __int8 ***)(a1 + 24);
          v52 = *(_QWORD *)(a1 + 8);
          if ( !(unsigned int)sub_BD2910((__int64)a2) )
          {
            if ( (v3[2] & 1) == 0
              || (v44 = *(_QWORD *)(v52 + 208),
                  v40 = sub_B43CB0((__int64)v3),
                  (v41 = sub_255ED30(*(_QWORD *)(v44 + 240), v40, 0)) != 0)
              && (unsigned __int8)sub_DFA750(v41) )
            {
              if ( v57 )
              {
                sub_256E5A0(v52, (__int64)a2, v47, v30, v31, v32);
                v7 = v57;
              }
              else
              {
                v61 = 257;
                v33 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
                v34 = v33;
                if ( v33 )
                {
                  v58 = v33;
                  sub_B51C90((__int64)v33, (__int64)v47, v29, (__int64)v60, 0, 0);
                  v34 = v58;
                }
                v59 = v34;
                sub_B44220(v34, (__int64)(v3 + 24), 0);
                sub_256E5A0(v52, (__int64)a2, v59, v35, v36, v37);
              }
              goto LABEL_15;
            }
          }
          goto LABEL_14;
        default:
          return 1;
      }
      if ( (v3[2] & 1) == 0
        || (v43 = *(_QWORD *)(v18 + 208),
            v38 = sub_B43CB0((__int64)v3),
            (v39 = sub_255ED30(*(_QWORD *)(v43 + 240), v38, 0)) != 0)
        && (unsigned __int8)sub_DFA750(v39) )
      {
        if ( v54 )
        {
          sub_256E5A0(v18, (__int64)a2, v46, v19, v20, v21);
          v7 = v54;
        }
        else
        {
          v61 = 257;
          v22 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
          v23 = v22;
          if ( v22 )
          {
            v55 = v22;
            sub_B51C90((__int64)v22, (__int64)v46, v51, (__int64)v60, 0, 0);
            v23 = v55;
          }
          v56 = v23;
          sub_B44220(v23, (__int64)(v3 + 24), 0);
          sub_256E5A0(v18, (__int64)a2, v56, v24, v25, v26);
        }
        goto LABEL_15;
      }
      goto LABEL_14;
    }
  }
  return 1;
}
