// Function: sub_69A8F0
// Address: 0x69a8f0
//
__int64 __fastcall sub_69A8F0(__int64 *a1, __int64 a2, __int16 a3, _DWORD *a4, _BYTE *a5)
{
  __int64 v7; // r12
  __int64 v8; // r15
  int v9; // ebx
  int v10; // eax
  __int64 result; // rax
  char v12; // dl
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // ebx
  bool v16; // zf
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rax
  char v23; // dl
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rax
  char v30; // dl
  __int64 v31; // rax
  __int64 v32; // rdi
  char v33; // dl
  __int64 v34; // rax
  unsigned __int8 v35; // al
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // r8
  char v43; // al
  __int64 v44; // rsi
  unsigned __int8 v45; // [rsp+8h] [rbp-48h]
  unsigned __int8 v46; // [rsp+8h] [rbp-48h]
  unsigned __int8 v49; // [rsp+18h] [rbp-38h]

  v7 = a2;
  v8 = *a1;
  *a5 = 0;
  if ( (unsigned int)sub_8D2930(a2) )
  {
    if ( (unsigned int)sub_8D2D50(v8) )
      goto LABEL_9;
    v9 = sub_8D26D0(v8);
    if ( v9 )
      goto LABEL_9;
    if ( (unsigned int)sub_8D2E30(v8) && *((_BYTE *)a1 + 16) == 2 && *((_BYTE *)a1 + 317) == 1 )
    {
      if ( !dword_4D04964 )
        goto LABEL_9;
      v49 = byte_4F07472[0];
      v41 = byte_4F07472[0];
      a2 = (unsigned int)sub_6E94D0();
      if ( v49 != 8 )
      {
        if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000001 && (_DWORD)a2 )
          sub_6E5C80(v41, a2, (char *)a1 + 68);
        goto LABEL_10;
      }
    }
    else
    {
      if ( (unsigned int)sub_8D3D40(v8) )
        goto LABEL_9;
      v12 = *(_BYTE *)(v8 + 140);
      if ( v12 == 12 )
      {
        v13 = v8;
        do
        {
          v13 = *(_QWORD *)(v13 + 160);
          v12 = *(_BYTE *)(v13 + 140);
        }
        while ( v12 == 12 );
      }
      if ( !v12 )
        goto LABEL_9;
      a2 = (unsigned int)sub_6E94D0();
    }
    if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000001 && (_DWORD)a2 )
    {
      v9 = 1;
      sub_6E5C80(8, a2, (char *)a1 + 68);
      goto LABEL_115;
    }
    goto LABEL_33;
  }
  v9 = 0;
  if ( (a3 & 2) == 0 )
  {
    if ( !dword_4F077C0 )
    {
LABEL_8:
      if ( !(unsigned int)sub_8D26D0(a2) && !(unsigned int)sub_8D3D40(a2) )
      {
        v30 = *(_BYTE *)(a2 + 140);
        if ( v30 == 12 )
        {
          v31 = a2;
          do
          {
            v31 = *(_QWORD *)(v31 + 160);
            v30 = *(_BYTE *)(v31 + 140);
          }
          while ( v30 == 12 );
        }
        if ( v30 )
        {
          a2 = unk_4D04000 == 0 ? 850 : 183;
          if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000001 )
          {
            v9 = 1;
            sub_6E5C80(8, a2, a4);
            goto LABEL_115;
          }
LABEL_33:
          v9 = 0;
LABEL_115:
          *a5 |= 1u;
          goto LABEL_10;
        }
      }
LABEL_9:
      v9 = 0;
      goto LABEL_10;
    }
    if ( (a3 & 0x200) == 0 && !(unsigned int)sub_6E97C0(a1) )
      goto LABEL_156;
  }
  if ( !(unsigned int)sub_8D2E30(a2) || !(unsigned int)sub_8D2930(v8) && !(unsigned int)sub_8D3D40(v8) )
  {
LABEL_156:
    if ( dword_4F077C0 && (unsigned int)sub_6E9880(a1) && (unsigned int)sub_8D2E30(a2) )
      goto LABEL_9;
    goto LABEL_8;
  }
  if ( !dword_4D04964 )
    goto LABEL_9;
  v35 = byte_4F07472[0];
  a2 = unk_4D04000 == 0 ? 850 : 183;
  if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000001 )
  {
    v46 = byte_4F07472[0];
    v9 = byte_4F07472[0] == 8;
    sub_6E5C80(byte_4F07472[0], a2, a4);
    v35 = v46;
  }
  if ( v35 == 8 )
    goto LABEL_115;
LABEL_10:
  if ( !(unsigned int)sub_8D2D50(v7) )
  {
    if ( (unsigned int)sub_8D2E30(v7) )
    {
      if ( (unsigned int)sub_8D2E30(v8) || (unsigned int)sub_8D2780(v8) || (unsigned int)sub_8D3D40(v8) )
        goto LABEL_12;
      if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) != 0x40000003 )
        goto LABEL_45;
      a2 = 44;
      sub_6E5C80(8, 44, (char *)a1 + 68);
    }
    else
    {
      if ( !(unsigned int)sub_8D3D10(v7) )
      {
        if ( (unsigned int)sub_8D26D0(v7) || (unsigned int)sub_8D3070(v7) )
          goto LABEL_12;
        if ( dword_4F077C0 )
        {
          if ( (unsigned int)sub_8D3A70(v7) )
          {
            if ( v8 == v7 )
              goto LABEL_12;
            a2 = v7;
            if ( (unsigned int)sub_8D97D0(v8, v7, 32, &dword_4F077C0, v42) )
              goto LABEL_12;
          }
          if ( dword_4F077C0 )
          {
            if ( *((_BYTE *)a1 + 16) == 2 )
            {
              if ( (unsigned int)sub_8D3B10(v7) )
              {
                a2 = v7;
                if ( sub_832ED0(a1, v7) )
                  goto LABEL_12;
              }
            }
          }
        }
        if ( (unsigned int)sub_8D3D40(v7) )
          goto LABEL_12;
        v23 = *(_BYTE *)(v7 + 140);
        if ( v23 == 12 )
        {
          v24 = v7;
          do
          {
            v24 = *(_QWORD *)(v24 + 160);
            v23 = *(_BYTE *)(v24 + 140);
          }
          while ( v23 == 12 );
        }
        if ( !v23 )
          goto LABEL_12;
        a2 = unk_4D04000 == 0 ? 851 : 184;
        if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000003 )
        {
          v9 = 1;
          sub_6E5C80(8, a2, a4);
        }
        goto LABEL_45;
      }
      if ( (unsigned int)sub_8D3D10(v8) || (unsigned int)sub_8D2780(v8) || (unsigned int)sub_8D3D40(v8) )
        goto LABEL_12;
      if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) != 0x40000003 )
        goto LABEL_45;
      a2 = 380;
      sub_6E5C80(8, 380, (char *)a1 + 68);
    }
    goto LABEL_151;
  }
  if ( (unsigned int)sub_8D2D50(v8) )
  {
LABEL_12:
    v10 = *(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF;
    goto LABEL_13;
  }
  if ( !(unsigned int)sub_8D2E30(v8) || !(unsigned int)sub_8D2780(v7) )
  {
    if ( (unsigned int)sub_8D3D40(v8) )
      goto LABEL_12;
    v33 = *(_BYTE *)(v8 + 140);
    if ( v33 == 12 )
    {
      v34 = v8;
      do
      {
        v34 = *(_QWORD *)(v34 + 160);
        v33 = *(_BYTE *)(v34 + 140);
      }
      while ( v33 == 12 );
    }
    if ( !v33 )
      goto LABEL_12;
    a2 = (unsigned int)sub_6E94D0();
LABEL_107:
    if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) != 0x40000003 || !(_DWORD)a2 )
      goto LABEL_45;
    sub_6E5C80(8, a2, (char *)a1 + 68);
LABEL_151:
    v9 = 1;
LABEL_45:
    *a5 |= 2u;
    goto LABEL_12;
  }
  if ( !dword_4D04964 )
    goto LABEL_12;
  v45 = byte_4F07472[0];
  v32 = byte_4F07472[0];
  a2 = (unsigned int)sub_6E94D0();
  if ( v45 == 8 )
    goto LABEL_107;
  v10 = *(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF;
  if ( v10 == 1073741827 )
  {
    if ( !(_DWORD)a2 )
      goto LABEL_14;
    sub_6E5C80(v32, a2, (char *)a1 + 68);
    v10 = *(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF;
  }
LABEL_13:
  if ( v10 != 1073741826 )
    goto LABEL_14;
  if ( (unsigned int)sub_8D2930(v7) )
  {
    if ( (unsigned int)sub_8D2D50(v8) )
      goto LABEL_14;
    if ( !(unsigned int)sub_8D2E30(v8)
      || *((_BYTE *)a1 + 16) != 2
      || (v43 = *((_BYTE *)a1 + 317), v43 != 12) && v43 != 1 )
    {
      if ( (unsigned int)sub_8D3D40(v8) )
        goto LABEL_14;
      v28 = *(unsigned __int8 *)(v8 + 140);
      if ( (_BYTE)v28 == 12 )
      {
        v29 = v8;
        do
        {
          v29 = *(_QWORD *)(v29 + 160);
          v28 = *(unsigned __int8 *)(v29 + 140);
        }
        while ( (_BYTE)v28 == 12 );
      }
      if ( (_BYTE)v28 && (unsigned int)sub_6E5430(v8, a2, v28, v25, v26, v27) )
      {
        v9 = 1;
        sub_6851C0(0x21Fu, (_DWORD *)a1 + 17);
        goto LABEL_14;
      }
      goto LABEL_62;
    }
    if ( dword_4D04964 )
    {
      v44 = (unsigned int)sub_6E94D0();
      sub_6E5C80(byte_4F07472[0], v44, (char *)a1 + 68);
      v9 = byte_4F07472[0] == 8;
    }
  }
  else
  {
    v17 = dword_4D04800;
    if ( dword_4D04800 && !dword_4D04964 && (unsigned int)sub_8D2A90(v7) )
    {
      if ( !(unsigned int)sub_8D2D50(v8) && !(unsigned int)sub_8D3D40(v8) )
      {
        v39 = *(unsigned __int8 *)(v8 + 140);
        if ( (_BYTE)v39 == 12 )
        {
          v40 = v8;
          do
          {
            v40 = *(_QWORD *)(v40 + 160);
            v39 = *(unsigned __int8 *)(v40 + 140);
          }
          while ( (_BYTE)v39 == 12 );
        }
        if ( (_BYTE)v39 )
        {
          v9 = 1;
          sub_69A8C0(543, (_DWORD *)a1 + 17, v39, v36, v37, v38);
          goto LABEL_14;
        }
        goto LABEL_62;
      }
    }
    else if ( (!(unsigned int)sub_8D2E30(v7) && !(unsigned int)sub_8D3D10(v7)
            || *((_BYTE *)a1 + 16) != 2
            || !(unsigned int)sub_712690(a1 + 18))
           && !(unsigned int)sub_8D26D0(v7)
           && (!dword_4F077BC
            || qword_4F077A8 > 0x76BFu
            || !(unsigned int)sub_8D2E30(v7)
            || !(unsigned int)sub_8D2E30(v8) && !(unsigned int)sub_8D2780(v8))
           && !(unsigned int)sub_8D3D40(v7) )
    {
      v21 = *(unsigned __int8 *)(v7 + 140);
      if ( (_BYTE)v21 == 12 )
      {
        v22 = v7;
        do
        {
          v22 = *(_QWORD *)(v22 + 160);
          v21 = *(unsigned __int8 *)(v22 + 140);
        }
        while ( (_BYTE)v21 == 12 );
      }
      if ( (_BYTE)v21 && (unsigned int)sub_6E5430(v7, v17, v21, v18, v19, v20) )
        sub_6851C0(0x369u, a4);
LABEL_62:
      v9 = 1;
    }
  }
LABEL_14:
  if ( !word_4D04898 )
    return v9 ^ 1u;
  if ( !(unsigned int)sub_8D2E30(v8) )
    return v9 ^ 1u;
  v14 = sub_8D46C0(v8);
  if ( !(unsigned int)sub_8D2600(v14) || !(unsigned int)sub_8D2EB0(v7) )
    return v9 ^ 1u;
  v15 = v9 ^ 1;
  v16 = (unsigned int)sub_6E91E0(28, a4) == 0;
  result = 0;
  if ( v16 )
    return v15;
  return result;
}
