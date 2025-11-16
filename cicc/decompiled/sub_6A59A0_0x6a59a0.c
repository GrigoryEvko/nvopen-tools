// Function: sub_6A59A0
// Address: 0x6a59a0
//
__int64 __fastcall sub_6A59A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 v9; // rsi
  char v10; // al
  __int64 v11; // r15
  __int64 v12; // rax
  int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rbx
  __int64 v20; // rax
  _BYTE *v21; // r13
  __int64 v22; // rdi
  char i; // al
  __int64 v24; // rdi
  char k; // al
  __int64 m; // r15
  char v27; // al
  __int64 v28; // rbx
  __int64 v29; // rdi
  char n; // al
  __int64 j; // r15
  __int64 ii; // r13
  __int64 v33; // rdx
  __int64 v34; // [rsp-10h] [rbp-1B0h]
  __int64 v35; // [rsp-8h] [rbp-1A8h]
  unsigned int v36; // [rsp+0h] [rbp-1A0h] BYREF
  int v37; // [rsp+4h] [rbp-19Ch] BYREF
  __int64 v38; // [rsp+8h] [rbp-198h] BYREF
  _QWORD v39[2]; // [rsp+10h] [rbp-190h] BYREF
  char v40; // [rsp+20h] [rbp-180h]
  __int64 v41; // [rsp+5Ch] [rbp-144h]
  __int64 v42; // [rsp+68h] [rbp-138h]
  _BYTE *v43; // [rsp+A0h] [rbp-100h]

  v37 = 0;
  if ( a1 )
  {
    sub_6F8AB0((_DWORD)a1, (unsigned int)v39, 0, 0, (unsigned int)&v38, (unsigned int)&v36, 0);
    v8 = v34;
    v9 = v35;
  }
  else
  {
    v38 = *(_QWORD *)&dword_4F063F8;
    v36 = dword_4F06650[0];
    sub_7B8B50(0, a2, a3, a4);
    v9 = 0;
    a1 = v39;
    sub_69ED20((__int64)v39, 0, 18, 0);
  }
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
  {
    v10 = *(_BYTE *)(qword_4D03C50 + 16LL);
    if ( v10 )
    {
      if ( v10 == 1 )
      {
        if ( (unsigned int)sub_6E5430(a1, v9, v5, v8, v6, v7) )
          sub_6851C0(0x3Cu, &v38);
      }
      else
      {
        if ( v10 != 2 )
          goto LABEL_7;
        if ( (unsigned int)sub_6E5430(a1, v9, v5, v8, v6, v7) )
          sub_6851C0(0x211u, &v38);
      }
    }
    else if ( (unsigned int)sub_6E5430(a1, v9, v5, v8, v6, v7) )
    {
      sub_6851C0(0x3Au, &v38);
    }
    sub_6E6260(a2);
    sub_6E6450(v39);
    goto LABEL_10;
  }
LABEL_7:
  v11 = v39[0];
  if ( (unsigned int)sub_8D32E0(v39[0]) )
  {
    v11 = sub_8D46C0(v11);
    if ( dword_4F077C4 != 2 )
      goto LABEL_9;
  }
  else if ( dword_4F077C4 != 2 )
  {
    goto LABEL_9;
  }
  if ( (unsigned int)sub_8E3200(v11) )
    sub_84EC30(7, 1, 0, 1, 0, (unsigned int)v39, 0, (__int64)&v38, v36, 0, 0, a2, 0, 0, (__int64)&v37);
LABEL_9:
  if ( v37 )
    goto LABEL_10;
  sub_6F69D0(v39, 0);
  if ( !(unsigned int)sub_8DBE70(v39[0]) && !(unsigned int)sub_6FB4D0(v39, 75) )
  {
    sub_6E6260(a2);
    goto LABEL_10;
  }
  v14 = sub_8D2EF0(v39[0]);
  v15 = v39[0];
  if ( v14 )
  {
    v16 = sub_8D46C0(v39[0]);
  }
  else if ( (unsigned int)sub_8DBE70(v39[0]) )
  {
    v16 = *(_QWORD *)&dword_4D03B80;
  }
  else
  {
    v16 = sub_72C930(v15);
  }
  v17 = sub_6F6F40(v39, 0);
  v18 = a2;
  v19 = sub_73DC30(3, v16, v17);
  sub_6E7150(v19, a2);
  if ( (unsigned int)sub_8D2600(v16) )
  {
    if ( dword_4F077C4 == 2 )
    {
      v18 = (__int64)v39;
      sub_6E68E0(852, v39);
      sub_6E6840(a2);
    }
    else if ( (*(_BYTE *)(v16 + 140) & 0xFB) != 8 || (v18 = 1, !(unsigned int)sub_8D4C10(v16, 1)) )
    {
      if ( !dword_4F077C0 )
      {
        *(_BYTE *)(a2 + 17) = 2;
        *(_BYTE *)(v19 + 25) &= ~1u;
      }
    }
  }
  if ( dword_4F04C58 != -1 )
  {
    v20 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
    if ( v20 )
    {
      if ( (*(_BYTE *)(v20 + 198) & 0x10) != 0 && (!qword_4D03C50 || *(char *)(qword_4D03C50 + 18LL) >= 0) && v40 == 1 )
      {
        v21 = v43;
        if ( v43 )
        {
          v22 = *(_QWORD *)v43;
          for ( i = *(_BYTE *)(*(_QWORD *)v43 + 140LL); i == 12; i = *(_BYTE *)(v22 + 140) )
            v22 = *(_QWORD *)(v22 + 160);
          if ( i != 6 )
            goto LABEL_87;
          for ( j = sub_8D46C0(v22); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          if ( (unsigned int)sub_8D2FF0(j, v18) || (unsigned int)sub_8D3030(j) )
          {
            v27 = v21[24];
            if ( v27 == 1 )
            {
              if ( v21[56] != 5 )
              {
LABEL_47:
                sub_6851C0(0xDE5u, &v38);
                goto LABEL_48;
              }
              v21 = (_BYTE *)sub_6E8430(v21);
              v27 = v21[24];
            }
          }
          else
          {
LABEL_87:
            if ( v21[24] != 1 || v21[56] != 5 )
              goto LABEL_48;
            v21 = (_BYTE *)sub_6E8430(v21);
            v24 = *(_QWORD *)v21;
            for ( k = *(_BYTE *)(*(_QWORD *)v21 + 140LL); k == 12; k = *(_BYTE *)(v24 + 140) )
              v24 = *(_QWORD *)(v24 + 160);
            if ( k != 6 )
              goto LABEL_48;
            for ( m = sub_8D46C0(v24); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
              ;
            if ( !(unsigned int)sub_8D2FF0(m, v18) && !(unsigned int)sub_8D3030(m) )
              goto LABEL_48;
            v27 = v21[24];
          }
          if ( v27 == 3 )
          {
            v28 = *((_QWORD *)v21 + 7);
            if ( v28 )
            {
              v29 = *(_QWORD *)(v28 + 120);
              for ( n = *(_BYTE *)(v29 + 140); n == 12; n = *(_BYTE *)(v29 + 140) )
                v29 = *(_QWORD *)(v29 + 160);
              if ( n == 6 )
              {
                for ( ii = sub_8D46C0(v29); *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
                  ;
                if ( (unsigned int)sub_8D2FF0(ii, v18) || (unsigned int)sub_8D3030(ii) )
                {
                  v33 = *(_QWORD *)(v28 + 8);
                  if ( v33 )
                  {
                    sub_6851A0(0xDE4u, &v38, v33);
                    goto LABEL_48;
                  }
                }
              }
            }
          }
          goto LABEL_47;
        }
      }
    }
  }
LABEL_48:
  if ( *(_BYTE *)(a2 + 17) == 1 && !(unsigned int)sub_6ED0A0(a2) )
    *(_QWORD *)(a2 + 88) = v42;
LABEL_10:
  *(_DWORD *)(a2 + 68) = v38;
  *(_WORD *)(a2 + 72) = WORD2(v38);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  v12 = v41;
  *(_QWORD *)(a2 + 76) = v41;
  unk_4F061D8 = v12;
  sub_6E3280(a2, &v38);
  sub_6E3BA0(a2, &v38, v36, 0);
  return sub_6E26D0(1, a2);
}
