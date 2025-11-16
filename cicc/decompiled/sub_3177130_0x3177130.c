// Function: sub_3177130
// Address: 0x3177130
//
void __fastcall sub_3177130(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int8 *v3; // r15
  int v4; // edx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  int v9; // r12d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int8 *v13; // r13
  __int64 v14; // r12
  int v15; // edx
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  int v22; // edx
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  _DWORD *v29; // rax
  unsigned __int8 *v30; // rax
  _QWORD *v31; // r14
  int v32; // eax
  _QWORD *v33; // r12
  unsigned __int8 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rdx
  unsigned __int8 *v37; // [rsp+0h] [rbp-A0h]
  __int64 v38; // [rsp+8h] [rbp-98h]
  __int64 v39; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+20h] [rbp-80h]
  int v41; // [rsp+20h] [rbp-80h]
  __int64 v42; // [rsp+20h] [rbp-80h]
  int v43; // [rsp+20h] [rbp-80h]
  __int64 v44; // [rsp+20h] [rbp-80h]
  __int64 v45; // [rsp+28h] [rbp-78h]
  __int64 v46; // [rsp+38h] [rbp-68h]
  const char *v47; // [rsp+40h] [rbp-60h] BYREF
  int v48; // [rsp+50h] [rbp-50h]
  __int16 v49; // [rsp+60h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 16);
  if ( v2 )
  {
    while ( 1 )
    {
      v3 = *(unsigned __int8 **)(v2 + 24);
      if ( *v3 == 85 && (unsigned __int8)sub_2A64220(*(__int64 **)a1, *((_QWORD *)v3 + 5)) )
      {
        v4 = *v3;
        if ( v4 == 40 )
        {
          v5 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v3);
        }
        else
        {
          v5 = -32;
          if ( v4 != 85 )
          {
            if ( v4 != 34 )
LABEL_77:
              BUG();
            v5 = -96;
          }
        }
        if ( (v3[7] & 0x80u) != 0 )
        {
          v6 = sub_BD2BC0((__int64)v3);
          v8 = v6 + v7;
          if ( (v3[7] & 0x80u) == 0 )
          {
            if ( (unsigned int)(v8 >> 4) )
LABEL_72:
              BUG();
          }
          else if ( (unsigned int)((v8 - sub_BD2BC0((__int64)v3)) >> 4) )
          {
            if ( (v3[7] & 0x80u) == 0 )
              goto LABEL_72;
            v9 = *(_DWORD *)(sub_BD2BC0((__int64)v3) + 8);
            if ( (v3[7] & 0x80u) == 0 )
              BUG();
            v10 = sub_BD2BC0((__int64)v3);
            v5 -= 32LL * (unsigned int)(*(_DWORD *)(v10 + v11 - 4) - v9);
          }
        }
        v37 = &v3[v5];
        v12 = *((_DWORD *)v3 + 1) & 0x7FFFFFF;
        v13 = &v3[-32 * v12];
        if ( v13 != v37 )
          break;
      }
LABEL_3:
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        return;
    }
    while ( 1 )
    {
      v14 = (v13 - &v3[-32 * v12]) >> 5;
      v38 = (unsigned int)v14;
      v15 = *v3;
      v39 = *(_QWORD *)&v3[32 * ((unsigned int)v14 - v12)];
      v45 = *(_QWORD *)(v39 + 8);
      if ( v15 == 40 )
      {
        v16 = 32LL * (unsigned int)sub_B491D0((__int64)v3);
      }
      else
      {
        v16 = 0;
        if ( v15 != 85 )
        {
          if ( v15 != 34 )
            goto LABEL_77;
          v16 = 64;
        }
      }
      if ( (v3[7] & 0x80u) == 0 )
        goto LABEL_61;
      v17 = sub_BD2BC0((__int64)v3);
      v40 = v18 + v17;
      if ( (v3[7] & 0x80u) == 0 )
        break;
      if ( !(unsigned int)((v40 - sub_BD2BC0((__int64)v3)) >> 4) )
        goto LABEL_61;
      if ( (v3[7] & 0x80u) == 0 )
        goto LABEL_75;
      v41 = *(_DWORD *)(sub_BD2BC0((__int64)v3) + 8);
      if ( (v3[7] & 0x80u) == 0 )
        BUG();
      v19 = sub_BD2BC0((__int64)v3);
      v21 = 32LL * (unsigned int)(*(_DWORD *)(v19 + v20 - 4) - v41);
LABEL_29:
      if ( (unsigned int)v14 < (unsigned int)((32LL * (*((_DWORD *)v3 + 1) & 0x7FFFFFF) - 32 - v16 - v21) >> 5)
        && (unsigned __int8)sub_B49B80((__int64)v3, v14, 81) )
      {
        goto LABEL_43;
      }
      v22 = *v3;
      if ( v22 == 40 )
      {
        v23 = 32LL * (unsigned int)sub_B491D0((__int64)v3);
      }
      else
      {
        v23 = 0;
        if ( v22 != 85 )
        {
          if ( v22 != 34 )
            goto LABEL_77;
          v23 = 64;
        }
      }
      if ( (v3[7] & 0x80u) == 0 )
        goto LABEL_63;
      v24 = sub_BD2BC0((__int64)v3);
      v42 = v25 + v24;
      if ( (v3[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v42 >> 4) )
LABEL_73:
          BUG();
LABEL_63:
        v28 = 0;
        goto LABEL_41;
      }
      if ( !(unsigned int)((v42 - sub_BD2BC0((__int64)v3)) >> 4) )
        goto LABEL_63;
      if ( (v3[7] & 0x80u) == 0 )
        goto LABEL_73;
      v43 = *(_DWORD *)(sub_BD2BC0((__int64)v3) + 8);
      if ( (v3[7] & 0x80u) == 0 )
        BUG();
      v26 = sub_BD2BC0((__int64)v3);
      v28 = 32LL * (unsigned int)(*(_DWORD *)(v26 + v27 - 4) - v43);
LABEL_41:
      if ( (unsigned int)v14 >= (unsigned int)((32LL * (*((_DWORD *)v3 + 1) & 0x7FFFFFF) - 32 - v23 - v28) >> 5) )
      {
        v29 = (_DWORD *)sub_B49810((__int64)v3, v14);
        if ( *(_DWORD *)(*(_QWORD *)v29 + 8LL)
          || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)&v3[32 * (unsigned int)(v14 - v29[2])
                                                 + 32
                                                 * ((unsigned int)v29[2]
                                                  - (unsigned __int64)(*((_DWORD *)v3 + 1) & 0x7FFFFFF))]
                                  + 8LL)
                      + 8LL) != 14 )
        {
LABEL_48:
          if ( !sub_CF49B0(v3, v14, 50) || *(_BYTE *)(v45 + 8) != 14 )
            goto LABEL_44;
          goto LABEL_50;
        }
      }
      else if ( !(unsigned __int8)sub_B49B80((__int64)v3, v14, 51) )
      {
        goto LABEL_48;
      }
LABEL_43:
      if ( *(_BYTE *)(v45 + 8) != 14 )
        goto LABEL_44;
LABEL_50:
      v30 = sub_31770C0((__int64 **)a1, v3, (unsigned __int8 *)v39);
      if ( v30 )
      {
        v31 = (_QWORD *)*((_QWORD *)v30 + 1);
        v44 = (__int64)v30;
        v32 = *(_DWORD *)(a1 + 792);
        v47 = "specialized.arg.";
        *(_DWORD *)(a1 + 792) = ++v32;
        v48 = v32;
        v49 = 2307;
        BYTE4(v46) = 0;
        v33 = sub_BD2C40(88, unk_3F0FAE8);
        if ( v33 )
          sub_B30000((__int64)v33, *(_QWORD *)(a1 + 8), v31, 1, 7, v44, (__int64)&v47, 0, 0, v46, 0);
        v34 = &v3[32 * (v38 - (*((_DWORD *)v3 + 1) & 0x7FFFFFF))];
        if ( *(_QWORD *)v34 )
        {
          v35 = *((_QWORD *)v34 + 1);
          **((_QWORD **)v34 + 2) = v35;
          if ( v35 )
            *(_QWORD *)(v35 + 16) = *((_QWORD *)v34 + 2);
        }
        *(_QWORD *)v34 = v33;
        if ( v33 )
        {
          v36 = v33[2];
          *((_QWORD *)v34 + 1) = v36;
          if ( v36 )
            *(_QWORD *)(v36 + 16) = v34 + 8;
          *((_QWORD *)v34 + 2) = v33 + 2;
          v33[2] = v34;
        }
      }
LABEL_44:
      v13 += 32;
      if ( v37 == v13 )
        goto LABEL_3;
      v12 = *((_DWORD *)v3 + 1) & 0x7FFFFFF;
    }
    if ( (unsigned int)(v40 >> 4) )
LABEL_75:
      BUG();
LABEL_61:
    v21 = 0;
    goto LABEL_29;
  }
}
