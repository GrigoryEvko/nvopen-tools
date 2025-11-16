// Function: sub_632A80
// Address: 0x632a80
//
__int64 __fastcall sub_632A80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6, __int64 a7)
{
  char v10; // al
  char v11; // r12
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 i; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 v21; // r10
  __int64 j; // rax
  __int64 v23; // rax
  __int64 result; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // r14
  __int64 v29; // r12
  char k; // al
  __int64 v31; // r15
  int v32; // eax
  __int64 v33; // r10
  int v34; // eax
  __int64 n; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 m; // rax
  __int64 v39; // rax
  __int64 v40; // r10
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r9
  __int64 v44; // r10
  __int64 v45; // rax
  __int64 v46; // r11
  __int64 v47; // r10
  __int64 v48; // r8
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 v53; // [rsp+8h] [rbp-98h]
  __int64 v54; // [rsp+10h] [rbp-90h]
  __int64 v55; // [rsp+18h] [rbp-88h]
  __int64 v56; // [rsp+18h] [rbp-88h]
  __int64 v57; // [rsp+20h] [rbp-80h]
  __int64 v58; // [rsp+20h] [rbp-80h]
  __int64 v59; // [rsp+20h] [rbp-80h]
  __int64 v60; // [rsp+20h] [rbp-80h]
  __int64 v61; // [rsp+20h] [rbp-80h]
  __int64 v62; // [rsp+20h] [rbp-80h]
  __int64 v63; // [rsp+28h] [rbp-78h]
  __int64 v64; // [rsp+28h] [rbp-78h]
  char v65; // [rsp+30h] [rbp-70h]
  __int64 v66; // [rsp+30h] [rbp-70h]
  bool v67; // [rsp+47h] [rbp-59h]
  _BYTE v69[64]; // [rsp+60h] [rbp-40h] BYREF

  v10 = a5[43];
  v11 = *(_BYTE *)(a2 + 140);
  v67 = (v10 & 0x40) != 0;
  a5[43] = v10 | 0x40;
  sub_716470(a1, v69);
  if ( a4 )
  {
    v63 = a3;
    v65 = v11;
    v12 = a4;
    while ( 1 )
    {
      v13 = *(_QWORD *)(v12 + 40);
      for ( i = v13; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v15 = *(_QWORD *)(*(_QWORD *)i + 96LL);
      if ( (*(_BYTE *)(v15 + 176) & 1) != 0 || !*(_QWORD *)(v15 + 16) && *(_QWORD *)(v15 + 8) )
        break;
      if ( dword_4D048B8 && (*(_BYTE *)(v15 + 177) & 2) == 0 )
        break;
      if ( (unsigned int)sub_8D32E0(*(_QWORD *)(v12 + 40)) || (unsigned int)sub_630FC0(v13) )
      {
        a5[41] |= 8u;
        if ( (a5[40] & 0x40) == 0 )
          goto LABEL_49;
      }
      else if ( (a5[40] & 0x40) == 0 )
      {
LABEL_49:
        v25 = sub_724D50(10);
        *(_QWORD *)(v25 + 128) = v13;
        v57 = v25;
        v26 = sub_72FD90(*(_QWORD *)(v13 + 160), 3);
        v21 = v57;
        if ( v26 )
        {
          a5[41] |= 0x10u;
          *(_BYTE *)(v57 + 170) |= 0x60u;
        }
        else if ( *(_QWORD *)(*(_QWORD *)(v13 + 168) + 8LL) )
        {
          a5[41] |= 0x10u;
          *(_BYTE *)(v57 + 170) |= 0x60u;
        }
        goto LABEL_23;
      }
LABEL_11:
      v12 = *(_QWORD *)(v12 + 8);
      if ( !v12 )
      {
        v11 = v65;
        a3 = v63;
        goto LABEL_13;
      }
    }
    v21 = sub_6354B0(*(_QWORD *)(v12 + 40), a5, a6);
LABEL_23:
    if ( (a5[40] & 0x40) == 0 )
    {
      *(_WORD *)(v21 + 170) |= 0x8080u;
      sub_72A690(v21, a1, v12, 0);
    }
    goto LABEL_11;
  }
LABEL_13:
  v16 = sub_72FD90(a3, 7);
  v66 = v16;
  if ( v16 == a7 )
    goto LABEL_46;
  v18 = v16;
  v19 = 0;
  do
  {
    if ( (*(_BYTE *)(v18 + 145) & 0x20) != 0 )
    {
      if ( v11 == 11 )
      {
        v27 = a7;
        if ( a7 )
          goto LABEL_120;
        v19 = v18;
        v18 = sub_72FD90(*(_QWORD *)(v18 + 112), 7);
        goto LABEL_53;
      }
LABEL_16:
      v19 = v18;
      goto LABEL_17;
    }
    v20 = *(_QWORD *)(v18 + 120);
    if ( (unsigned int)sub_8D32E0(v20) || (unsigned int)sub_630FC0(v20) )
    {
      a5[41] |= 8u;
      goto LABEL_17;
    }
    if ( (unsigned int)sub_8D3410(v20) )
    {
      v20 = sub_8D40F0(v20);
      if ( *(_BYTE *)(v20 + 140) != 12 )
        goto LABEL_30;
      goto LABEL_28;
    }
    while ( *(_BYTE *)(v20 + 140) == 12 )
LABEL_28:
      v20 = *(_QWORD *)(v20 + 160);
LABEL_30:
    if ( (unsigned int)sub_8D3AD0(v20) )
    {
      for ( j = v20; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      v23 = *(_QWORD *)(*(_QWORD *)j + 96LL);
      if ( (*(_BYTE *)(v23 + 176) & 1) != 0
        || !*(_QWORD *)(v23 + 16) && *(_QWORD *)(v23 + 8)
        || dword_4D048B8 && *(_QWORD *)(v23 + 24) && (*(_BYTE *)(v23 + 177) & 2) == 0 )
      {
        goto LABEL_16;
      }
      if ( (*(_BYTE *)(v23 + 178) & 0x40) == 0 && *(_BYTE *)(*(_QWORD *)(v20 + 168) + 113LL) == 2 )
        v19 = v18;
    }
LABEL_17:
    v18 = sub_72FD90(*(_QWORD *)(v18 + 112), 7);
  }
  while ( a7 != v18 );
  if ( !(a7 | v19) )
    goto LABEL_43;
  if ( a7 )
  {
    if ( v11 == 11 && v19 && (*(_BYTE *)(v19 + 145) & 0x20) != 0 )
    {
      v27 = a7;
      v18 = v19;
      goto LABEL_120;
    }
LABEL_60:
    v64 = v19;
    v28 = v66;
    while ( 2 )
    {
      v29 = *(_QWORD *)(v28 + 120);
      for ( k = *(_BYTE *)(v29 + 140); k == 12; k = *(_BYTE *)(v29 + 140) )
        v29 = *(_QWORD *)(v29 + 160);
      if ( (*(_BYTE *)(v28 + 145) & 0x20) != 0 )
      {
        v31 = 0;
        v33 = sub_6327C0(v28, *(_QWORD *)(v28 + 152), a2, a5, a6, v17);
      }
      else
      {
        v31 = 0;
        if ( k == 8 )
        {
          for ( m = sub_8D40F0(v29); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
            ;
          v31 = v29;
          v29 = m;
        }
        if ( !(unsigned int)sub_8D3AD0(v29) )
        {
LABEL_66:
          if ( (a5[40] & 0x40) == 0 )
            goto LABEL_67;
LABEL_73:
          v28 = sub_72FD90(*(_QWORD *)(v28 + 112), 7);
          if ( v28 == v18 )
          {
            v19 = v64;
            goto LABEL_75;
          }
          continue;
        }
        for ( n = v29; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
          ;
        if ( *(_BYTE *)(*(_QWORD *)(v29 + 168) + 113LL) == 2 )
        {
          v40 = 0;
          if ( (a5[40] & 0x40) == 0 )
          {
            v41 = sub_724D50(10);
            *(_BYTE *)(v41 + 170) |= 0x80u;
            v40 = v41;
            *(_QWORD *)(v41 + 128) = v29;
          }
          v60 = v40;
          v55 = *(_QWORD *)(v29 + 160);
          v42 = sub_72FD90(v55, 7);
          v33 = v60;
          if ( v42 )
          {
            while ( (*(_BYTE *)(v42 + 145) & 0x20) == 0 )
            {
              v42 = sub_72FD90(*(_QWORD *)(v42 + 112), 7);
              if ( !v42 )
                goto LABEL_111;
            }
            v44 = v60;
            v61 = v42;
            v54 = v44;
            v45 = sub_6327C0(v42, *(_QWORD *)(v42 + 152), v29, a5, a6, v43);
            v46 = v61;
            v47 = v54;
            v48 = v45;
            if ( (a5[40] & 0x40) != 0 )
              goto LABEL_73;
            if ( v61 != v55 )
            {
              v53 = v45;
              v56 = v61;
              v62 = sub_724D50(13);
              v49 = sub_72CBE0();
              *(_BYTE *)(v62 + 176) |= 1u;
              *(_QWORD *)(v62 + 184) = v56;
              *(_QWORD *)(v62 + 128) = v49;
              sub_72A690(v62, v54, 0, 0);
              v48 = v53;
              v46 = v56;
              v47 = v54;
            }
            *(_BYTE *)(v48 + 170) |= 0x80u;
            v60 = v47;
            sub_72A690(v48, v47, 0, v46);
LABEL_111:
            v33 = v60;
          }
        }
        else
        {
          v36 = *(_QWORD *)(*(_QWORD *)n + 96LL);
          if ( (*(_BYTE *)(v36 + 176) & 1) == 0
            && (*(_QWORD *)(v36 + 16) || !*(_QWORD *)(v36 + 8))
            && (!dword_4D048B8 || (*(_BYTE *)(v36 + 177) & 2) != 0) )
          {
            goto LABEL_66;
          }
          v37 = sub_6354B0(v29, a5, a6);
          v33 = v37;
          if ( v31 )
          {
            if ( (a5[40] & 0x40) != 0 )
              goto LABEL_73;
            v33 = sub_62FF50(v37, v31);
          }
        }
      }
      break;
    }
    if ( (a5[40] & 0x40) != 0 )
      goto LABEL_73;
    if ( !v33 )
    {
LABEL_67:
      v58 = sub_724D50(10);
      v32 = sub_8D3350(v29);
      v33 = v58;
      if ( v32 )
      {
        if ( v31 )
          goto LABEL_69;
        sub_72BB40(v29, v58);
        v33 = v58;
      }
      else
      {
        if ( v31 )
        {
LABEL_69:
          *(_QWORD *)(v58 + 128) = v31;
          if ( (unsigned __int8)(*(_BYTE *)(v29 + 140) - 9) <= 2u )
          {
            v50 = sub_72FD90(*(_QWORD *)(v29 + 160), 3);
            v33 = v58;
            if ( !v50 && !*(_QWORD *)(*(_QWORD *)(v29 + 168) + 8LL) )
              goto LABEL_72;
          }
          v59 = v33;
          v34 = sub_8D4440(v31);
          v33 = v59;
          if ( v34 )
            goto LABEL_72;
        }
        else
        {
          *(_QWORD *)(v58 + 128) = v29;
          if ( (unsigned __int8)(*(_BYTE *)(v29 + 140) - 9) <= 2u )
          {
            v39 = sub_72FD90(*(_QWORD *)(v29 + 160), 3);
            v33 = v58;
            if ( !v39 && !*(_QWORD *)(*(_QWORD *)(v29 + 168) + 8LL) )
              goto LABEL_72;
          }
        }
        a5[41] |= 0x10u;
        *(_BYTE *)(v33 + 170) |= 0x60u;
      }
    }
LABEL_72:
    *(_BYTE *)(v33 + 170) |= 0x80u;
    sub_72A690(v33, a1, 0, v28);
    goto LABEL_73;
  }
  v18 = sub_72FD90(*(_QWORD *)(v19 + 112), 7);
  if ( v11 != 11 )
  {
LABEL_124:
    if ( v66 != v18 )
      goto LABEL_60;
LABEL_75:
    if ( !a7 )
    {
      if ( v19 )
        goto LABEL_77;
LABEL_43:
      if ( v66 )
        goto LABEL_44;
    }
    goto LABEL_46;
  }
LABEL_53:
  if ( (*(_BYTE *)(v19 + 145) & 0x20) != 0 )
  {
    v27 = v18;
    v18 = v19;
LABEL_120:
    if ( (a5[40] & 0x40) != 0 || *(_QWORD *)(a2 + 160) == v18 )
    {
      v66 = v18;
      v19 = v18;
      v18 = v27;
    }
    else
    {
      v19 = v18;
      v51 = sub_724D50(13);
      v52 = sub_72CBE0();
      *(_BYTE *)(v51 + 170) |= 0x80u;
      *(_BYTE *)(v51 + 176) |= 1u;
      *(_QWORD *)(v51 + 184) = v18;
      *(_QWORD *)(v51 + 128) = v52;
      sub_72A690(v51, a1, 0, 0);
      v66 = v18;
      v18 = v27;
    }
    goto LABEL_124;
  }
  v66 = sub_72FD90(*(_QWORD *)(a2 + 160), 7);
  v18 = sub_72FD90(*(_QWORD *)(v66 + 112), 7);
  if ( v66 != v18 )
    goto LABEL_60;
LABEL_77:
  if ( sub_72FD90(*(_QWORD *)(v19 + 112), 7) )
  {
LABEL_44:
    a5[41] |= 0x10u;
    if ( a1 )
      *(_BYTE *)(a1 + 170) |= 0x60u;
  }
LABEL_46:
  sub_716490(v69);
  result = (v67 << 6) | a5[43] & 0xBFu;
  a5[43] = (v67 << 6) | a5[43] & 0xBF;
  return result;
}
