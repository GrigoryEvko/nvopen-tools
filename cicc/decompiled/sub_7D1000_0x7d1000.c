// Function: sub_7D1000
// Address: 0x7d1000
//
__int64 __fastcall sub_7D1000(__int64 a1, __int64 a2, __int64 a3, int *a4)
{
  __int64 v4; // r15
  _QWORD *v6; // r13
  _QWORD *v9; // r14
  __int64 v10; // rax
  char v11; // dl
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rsi
  char v15; // al
  __int64 v16; // rbx
  __int64 v17; // rcx
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rsi
  char v21; // dl
  char v22; // al
  _BOOL4 v23; // eax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // eax
  unsigned __int64 v27; // rdx
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rax
  char v33; // dl
  int v34; // eax
  __int64 v35; // rax
  _QWORD *v36; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+10h] [rbp-60h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+10h] [rbp-60h]
  __int64 v40; // [rsp+18h] [rbp-58h]
  __int64 v41; // [rsp+20h] [rbp-50h]
  bool v42; // [rsp+2Bh] [rbp-45h]
  int v43; // [rsp+2Ch] [rbp-44h]
  _DWORD v44[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v4 = a2;
  v42 = a2 != 0;
  if ( !dword_4F077BC || !a2 )
  {
    v43 = 0;
LABEL_7:
    v6 = *(_QWORD **)(a1 + 544);
    if ( !v6 )
      return v4;
    goto LABEL_8;
  }
  v43 = *a4;
  if ( !*a4 )
    goto LABEL_7;
  v6 = *(_QWORD **)(a1 + 544);
  v43 = *(_BYTE *)(a2 + 80) == 23;
  if ( v6 )
  {
LABEL_8:
    v41 = a2;
    v9 = v6;
    v37 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !a4[22] && !a4[6] || (*(_BYTE *)(v9[2] + 40LL) & 0x20) != 0 )
        {
          v40 = v9[3];
          v10 = *(_QWORD *)(v40 + 160);
          v11 = *(_BYTE *)(v10 + 80);
          if ( v11 == 16 )
          {
            v10 = **(_QWORD **)(v10 + 88);
            v11 = *(_BYTE *)(v10 + 80);
          }
          if ( v11 == 24 )
            v10 = *(_QWORD *)(v10 + 88);
          v12 = *(_QWORD *)(v10 + 88);
          if ( (*(_BYTE *)(v12 + 124) & 1) != 0 )
            v12 = sub_735B70(v12);
          v13 = *(_QWORD *)(v12 + 128);
          v14 = *(_QWORD *)a3;
          if ( unk_4D03F98 )
          {
            if ( v13 )
            {
              if ( *(_QWORD *)(v14 + 64) )
              {
                v15 = *(_BYTE *)(v13 + 28);
                if ( !v15 || v15 == 3 )
                {
                  sub_824D70(v13);
                  v14 = *(_QWORD *)a3;
                }
              }
            }
          }
          v16 = sub_883800(v40, v14);
          if ( v16 )
            break;
        }
        v9 = (_QWORD *)v9[1];
        if ( !v9 )
          return v41;
      }
      v17 = v37;
      v36 = v9;
      v18 = v41;
      do
      {
        while ( 1 )
        {
          v19 = *(unsigned __int8 *)(v16 + 80);
          v44[0] = 0;
          v20 = v16;
          v21 = v19;
          if ( (_BYTE)v19 == 16 )
          {
            v20 = **(_QWORD **)(v16 + 88);
            v21 = *(_BYTE *)(v20 + 80);
          }
          if ( v21 == 24 )
            v20 = *(_QWORD *)(v20 + 88);
          if ( dword_4F04BA0[v19] != a4[31] )
            goto LABEL_26;
          v22 = *(_BYTE *)(v20 + 83);
          if ( (v22 & 0x40) == 0 && (*(_BYTE *)(v16 + 83) & 0x40) == 0 )
          {
LABEL_34:
            if ( v22 < 0 )
              goto LABEL_50;
            goto LABEL_35;
          }
          if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || (unsigned __int64)(qword_4F077A8 - 50000LL) > 0x270F )
            goto LABEL_48;
          v33 = *(_BYTE *)(v20 + 80);
          if ( v33 == 20 )
            goto LABEL_34;
          if ( v33 != 17 || (v39 = v17, v34 = sub_8780F0(v20), v17 = v39, !v34) )
          {
LABEL_48:
            if ( !a4[6] && !a4[8] )
              goto LABEL_26;
          }
          if ( *(char *)(v20 + 83) < 0 )
          {
LABEL_50:
            if ( unk_4F04C48 != -1 )
            {
              v25 = *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 368);
              if ( v20 == v25 )
              {
                if ( v25 )
                  goto LABEL_26;
              }
            }
          }
LABEL_35:
          v38 = v17;
          v23 = sub_7CF0D0(v16, v20, a4);
          v17 = v38;
          if ( !v23 || v43 && *(_BYTE *)(v20 + 80) != 23 )
            goto LABEL_26;
          if ( !unk_4D047BC )
            break;
          v26 = a4[33];
          if ( v26 >= *(_DWORD *)(v40 + 168) || !v26 || (*(_BYTE *)(a3 + 16) & 8) != 0 )
            break;
          v27 = *(unsigned __int8 *)(v16 + 80);
          if ( (unsigned __int8)v27 > 0x14u )
            goto LABEL_26;
          v28 = 1182720;
          if ( !_bittest64(&v28, v27) )
            goto LABEL_26;
          v29 = qword_4F077A8;
          if ( qword_4F077A8 > 0x9CA3u && v42 )
          {
            if ( (_BYTE)v27 == 19 || qword_4F077A8 > 0x9EFBu )
              goto LABEL_26;
            if ( (_BYTE)v27 == 3 )
            {
              if ( *(_BYTE *)(v16 + 104) )
              {
                v35 = *(_QWORD *)(v16 + 88);
                if ( (*(_BYTE *)(v35 + 177) & 0x10) != 0 )
                {
                  if ( *(_QWORD *)(*(_QWORD *)(v35 + 168) + 168LL) )
                    goto LABEL_26;
                }
              }
              goto LABEL_73;
            }
            if ( (unsigned __int8)(v27 - 20) <= 1u )
              goto LABEL_26;
            if ( (((_BYTE)v27 - 7) & 0xFD) == 0 )
            {
              v30 = *(_QWORD *)(v16 + 88);
              if ( v30 )
              {
                if ( (*(_BYTE *)(v30 + 170) & 0x10) != 0 && **(_QWORD **)(v30 + 216) )
                  goto LABEL_26;
              }
            }
            if ( (_BYTE)v27 != 17 )
              goto LABEL_73;
            v31 = sub_8780F0(v16);
            v17 = v38;
            if ( v31 )
              goto LABEL_26;
            v29 = qword_4F077A8;
          }
          if ( v29 <= 0x76BF )
            break;
LABEL_73:
          if ( (*(_BYTE *)(a3 + 18) & 1) == 0 )
          {
            if ( v17 )
              goto LABEL_40;
            goto LABEL_75;
          }
LABEL_26:
          v16 = *(_QWORD *)(v16 + 32);
          if ( !v16 )
            goto LABEL_41;
        }
        if ( v17 )
          goto LABEL_40;
LABEL_75:
        v32 = sub_7CF9D0(*(_QWORD *)a3, a4[30], 0, 0);
        if ( v18 )
          v18 = sub_7D09E0(v32, v18, a3, 0, 0, a4[30], v44);
        else
          v18 = v32;
LABEL_40:
        v24 = sub_7D09E0(v18, v16, a3, 0, 0, a4[30], v44);
        v16 = *(_QWORD *)(v16 + 32);
        v17 = v24;
        v18 = v24;
      }
      while ( v16 );
LABEL_41:
      v41 = v18;
      v37 = v17;
      v9 = (_QWORD *)v36[1];
      if ( !v9 )
        return v41;
    }
  }
  return v4;
}
