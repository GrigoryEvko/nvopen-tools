// Function: sub_A1B020
// Address: 0xa1b020
//
void __fastcall sub_A1B020(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        char a8)
{
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // rbx
  unsigned int v11; // r12d
  char v12; // dl
  __int64 v13; // rax
  unsigned __int8 v14; // dl
  unsigned __int8 *v15; // rbx
  __int64 v16; // r14
  __int64 v17; // r12
  __int64 v18; // r13
  unsigned __int8 *v19; // r14
  unsigned __int64 v20; // rsi
  char v21; // al
  __int64 v22; // rdx
  unsigned __int64 v23; // rsi
  __int64 v24; // rdx
  unsigned int v25; // esi
  unsigned __int64 v26; // rsi
  __int64 v27; // rdx
  char v28; // al
  __int64 v29; // rbx
  __int64 v30; // r9
  _QWORD *v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rbx
  __int64 v35; // rcx
  __int64 v36; // rax
  _BYTE *v37; // rbx
  __int64 v38; // r14
  _QWORD *v39; // rbx
  __int64 v40; // rax
  _QWORD *v41; // rax
  int v42; // r10d
  __int64 v43; // rdx
  __int64 v44; // [rsp+0h] [rbp-80h]
  int v45; // [rsp+0h] [rbp-80h]
  __int64 v46; // [rsp+8h] [rbp-78h]
  _QWORD *v47; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v52; // [rsp+38h] [rbp-48h]
  unsigned int v53; // [rsp+38h] [rbp-48h]
  _QWORD *v54; // [rsp+40h] [rbp-40h]
  unsigned int v55; // [rsp+48h] [rbp-38h]
  int v56; // [rsp+4Ch] [rbp-34h]

  LODWORD(v8) = 0;
  v9 = a1;
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 16LL * (a2 - 4));
  v55 = a6;
  v54 = (_QWORD *)v10;
  sub_A17B10(a1, a2, *(_DWORD *)(a1 + 56));
  v56 = *(_DWORD *)(v10 + 8);
  if ( a8 )
  {
    LODWORD(v8) = 1;
    if ( (*(_BYTE *)(*(_QWORD *)v10 + 8LL) & 1) == 0 )
      sub_A1AF70(a1, *(_QWORD *)v10, a7);
  }
  v11 = 0;
  v52 = (unsigned __int8 *)a5;
  if ( v56 != (_DWORD)v8 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = *v54 + 16LL * (unsigned int)v8;
        v14 = *(_BYTE *)(v13 + 8);
        if ( (v14 & 1) != 0 )
        {
LABEL_12:
          ++v11;
          goto LABEL_13;
        }
        v12 = (v14 >> 1) & 7;
        if ( v12 == 3 )
          break;
        if ( v12 == 5 )
        {
          if ( v52 )
          {
            sub_A18720(v9, a5, a6, 1);
            v52 = 0;
          }
          else
          {
            v29 = a4 - v11;
            sub_A17CC0(v9, a4 - v11, 6);
            v30 = a3 + 8LL * v11;
            if ( *(_DWORD *)(v9 + 48) )
            {
              v41 = *(_QWORD **)(v9 + 24);
              v42 = *(_DWORD *)(v9 + 52);
              v43 = v41[1];
              if ( (unsigned __int64)(v43 + 4) > v41[2] )
              {
                v45 = *(_DWORD *)(v9 + 52);
                v47 = *(_QWORD **)(v9 + 24);
                sub_C8D290(v47, v41 + 3, v43 + 4, 1);
                v41 = v47;
                v42 = v45;
                v30 = a3 + 8LL * v11;
                v43 = v47[1];
              }
              *(_DWORD *)(*v41 + v43) = v42;
              v41[1] += 4LL;
              *(_QWORD *)(v9 + 48) = 0;
            }
            v31 = *(_QWORD **)(v9 + 24);
            v32 = 8 * v29;
            v33 = v31[1];
            v34 = (8 * v29) >> 3;
            if ( (unsigned __int64)(v34 + v33) > v31[2] )
            {
              v44 = v32;
              v46 = v30;
              sub_C8D290(v31, v31 + 3, v34 + v33, 1);
              v32 = v44;
              v30 = v46;
              v33 = v31[1];
            }
            v35 = v33 + *v31;
            if ( v32 > 0 )
            {
              v36 = 0;
              do
              {
                *(_BYTE *)(v35 + v36) = *(_QWORD *)(v30 + 8 * v36);
                ++v36;
              }
              while ( v34 - v36 > 0 );
              v33 = v31[1];
            }
            v31[1] = v33 + v34;
            while ( 1 )
            {
              v37 = *(_BYTE **)(v9 + 32);
              v38 = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 8LL);
              if ( v37 && (unsigned __int8)sub_CB7440(*(_QWORD *)(v9 + 32)) )
              {
                if ( !(unsigned __int8)sub_CB7440(v37) )
                  BUG();
                LOBYTE(v38) = (*(__int64 (__fastcall **)(_BYTE *))(*(_QWORD *)v37 + 80LL))(v37)
                            + v37[32]
                            - v37[16]
                            + v38;
              }
              if ( (v38 & 3) == 0 )
                break;
              v39 = *(_QWORD **)(v9 + 24);
              v40 = v39[1];
              if ( (unsigned __int64)(v40 + 1) > v39[2] )
              {
                sub_C8D290(*(_QWORD *)(v9 + 24), v39 + 3, v40 + 1, 1);
                v40 = v39[1];
              }
              *(_BYTE *)(*v39 + v40) = 0;
              ++v39[1];
            }
          }
          goto LABEL_10;
        }
        v23 = *(_QWORD *)(a3 + 8LL * v11);
        switch ( v12 )
        {
          case 2:
            if ( *(_QWORD *)v13 )
            {
              ++v11;
              sub_A17DE0(v9, v23, *(_QWORD *)v13);
              goto LABEL_13;
            }
            goto LABEL_12;
          case 4:
            if ( (unsigned __int8)(v23 - 97) <= 0x19u )
            {
              LODWORD(v23) = (char)v23 - 97;
            }
            else if ( (unsigned __int8)(v23 - 65) <= 0x19u )
            {
              LODWORD(v23) = (char)v23 - 39;
            }
            else if ( (unsigned __int8)(v23 - 48) <= 9u )
            {
              LODWORD(v23) = (char)v23 + 4;
            }
            else if ( (_BYTE)v23 == 46 )
            {
              LODWORD(v23) = 62;
            }
            else
            {
              if ( (_BYTE)v23 != 95 )
LABEL_104:
                BUG();
              LODWORD(v23) = 63;
            }
            LODWORD(v24) = 6;
            break;
          case 1:
            v24 = *(_QWORD *)v13;
            if ( !*(_QWORD *)v13 )
            {
              ++v11;
              goto LABEL_13;
            }
            break;
          default:
LABEL_102:
            BUG();
        }
        ++v11;
        sub_A17B10(v9, v23, v24);
LABEL_13:
        LODWORD(v8) = v8 + 1;
        if ( v56 == (_DWORD)v8 )
          return;
      }
      v15 = v52;
      v8 = (unsigned int)(v8 + 1);
      v16 = *v54 + 16 * v8;
      if ( !v52 )
        break;
      sub_A17CC0(v9, v55, 6);
      if ( v55 )
      {
        v53 = v11;
        v17 = v9;
        v18 = v16;
        v19 = &v15[(unsigned int)a6];
        while ( 1 )
        {
          v20 = *v15;
          v21 = (*(_BYTE *)(v18 + 8) >> 1) & 7;
          if ( v21 == 2 )
            break;
          if ( v21 == 4 )
          {
            if ( (unsigned __int8)(v20 - 97) <= 0x19u )
            {
              v25 = (char)v20 - 97;
LABEL_49:
              sub_A17B10(v17, v25, 6);
              goto LABEL_27;
            }
            if ( (unsigned __int8)(v20 - 65) <= 0x19u )
            {
              v25 = (char)v20 - 39;
              goto LABEL_49;
            }
            if ( (unsigned __int8)(v20 - 48) <= 9u )
            {
              v25 = (char)v20 + 4;
              goto LABEL_49;
            }
            if ( (_BYTE)v20 == 46 )
            {
              v25 = 62;
              goto LABEL_49;
            }
            if ( (_BYTE)v20 != 95 )
              goto LABEL_104;
            LODWORD(v20) = 63;
            LODWORD(v22) = 6;
LABEL_26:
            sub_A17B10(v17, v20, v22);
LABEL_27:
            if ( v19 == ++v15 )
              goto LABEL_28;
          }
          else
          {
            if ( v21 != 1 )
              goto LABEL_102;
            v22 = *(_QWORD *)v18;
            if ( *(_QWORD *)v18 )
              goto LABEL_26;
LABEL_31:
            if ( v19 == ++v15 )
            {
LABEL_28:
              v9 = v17;
              v11 = v53;
              v52 = 0;
              goto LABEL_10;
            }
          }
        }
        if ( *(_QWORD *)v18 )
          sub_A17DE0(v17, v20, *(_QWORD *)v18);
        goto LABEL_31;
      }
      v52 = 0;
LABEL_10:
      LODWORD(v8) = v8 + 1;
      if ( v56 == (_DWORD)v8 )
        return;
    }
    sub_A17CC0(v9, a4 - v11, 6);
    if ( (_DWORD)a4 == v11 )
      goto LABEL_10;
    while ( 1 )
    {
      while ( 1 )
      {
        v26 = *(_QWORD *)(a3 + 8LL * v11);
        v28 = (*(_BYTE *)(v16 + 8) >> 1) & 7;
        if ( v28 == 2 )
        {
          if ( *(_QWORD *)v16 )
            sub_A17DE0(v9, v26, *(_QWORD *)v16);
          goto LABEL_64;
        }
        if ( v28 != 4 )
          break;
        if ( (unsigned __int8)(v26 - 97) <= 0x19u )
        {
          LODWORD(v26) = (char)v26 - 97;
        }
        else if ( (unsigned __int8)(v26 - 65) <= 0x19u )
        {
          LODWORD(v26) = (char)v26 - 39;
        }
        else if ( (unsigned __int8)(v26 - 48) <= 9u )
        {
          LODWORD(v26) = (char)v26 + 4;
        }
        else if ( (_BYTE)v26 == 46 )
        {
          LODWORD(v26) = 62;
        }
        else
        {
          if ( (_BYTE)v26 != 95 )
            goto LABEL_104;
          LODWORD(v26) = 63;
        }
        LODWORD(v27) = 6;
LABEL_63:
        sub_A17B10(v9, v26, v27);
LABEL_64:
        if ( (_DWORD)a4 == ++v11 )
          goto LABEL_10;
      }
      if ( v28 != 1 )
        goto LABEL_102;
      v27 = *(_QWORD *)v16;
      if ( *(_QWORD *)v16 )
        goto LABEL_63;
      if ( (_DWORD)a4 == ++v11 )
        goto LABEL_10;
    }
  }
}
