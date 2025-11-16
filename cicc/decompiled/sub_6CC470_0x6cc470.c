// Function: sub_6CC470
// Address: 0x6cc470
//
__int64 __fastcall sub_6CC470(unsigned __int8 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 v7; // rsi
  unsigned int v8; // r14d
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _DWORD *v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r15
  _QWORD *v18; // r14
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 i; // rdx
  int v23; // r15d
  bool v24; // [rsp+17h] [rbp-249h]
  _QWORD *v25; // [rsp+18h] [rbp-248h]
  __int64 v26; // [rsp+28h] [rbp-238h] BYREF
  _BYTE v27[160]; // [rsp+30h] [rbp-230h] BYREF
  _QWORD v28[2]; // [rsp+D0h] [rbp-190h] BYREF
  unsigned __int8 v29; // [rsp+E0h] [rbp-180h]
  __int64 v30; // [rsp+160h] [rbp-100h]

  if ( a3 )
  {
    v5 = *(_QWORD *)(a3 + 64);
    v6 = 0;
    if ( v5 )
      v6 = *(_QWORD *)(v5 + 16);
  }
  else
  {
    v5 = 0;
    v6 = 0;
  }
  sub_6E1DD0(&v26);
  v7 = (__int64)v27;
  sub_6E1E00(5, v27, 0, 1);
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(a2) )
    sub_8AE000(a2);
  if ( (unsigned int)sub_8D3410(a2) )
  {
    if ( !v6 && !(unsigned int)sub_8D23E0(a2) )
    {
      v7 = sub_8D40F0(a2);
      v8 = sub_6CC470(a1, v7, a3);
      goto LABEL_8;
    }
    goto LABEL_7;
  }
  if ( (unsigned int)sub_8D2310(a2) || (unsigned int)sub_8D2600(a2) || (v8 = sub_8D5830(a2)) != 0 )
  {
LABEL_7:
    v6 = 0;
    v8 = 0;
    goto LABEL_8;
  }
  if ( (unsigned int)sub_8D23B0(a2) )
  {
    if ( !dword_4F077BC || (v7 = (unsigned int)qword_4F077B4, (_DWORD)qword_4F077B4) )
    {
      v14 = v5 ? (_DWORD *)(v5 + 28) : &dword_4F077C8;
      if ( (unsigned int)sub_6E5430(dword_4F077BC, v7, v10, v11, v12, v13) )
      {
        v7 = (__int64)v14;
        v6 = 0;
        sub_6851C0(0x46u, v14);
        goto LABEL_8;
      }
    }
    goto LABEL_7;
  }
  v15 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v24 = (*(_BYTE *)(v15 + 7) & 8) != 0;
  if ( v6 )
  {
    v25 = 0;
    v16 = 0;
    do
    {
      v17 = *(_QWORD *)(v6 + 56);
      v18 = (_QWORD *)v16;
      v16 = sub_68B9A0(v17);
      if ( !v16 )
      {
        if ( dword_4F077BC )
        {
          v8 = qword_4F077B4;
          if ( !(_DWORD)qword_4F077B4 )
            goto LABEL_34;
        }
        v8 = sub_8D23B0(v17);
        if ( !v8 )
          goto LABEL_34;
        v8 = sub_8D3410(v17);
        if ( v8 )
        {
          v6 = (__int64)v25;
          v8 = 0;
          goto LABEL_35;
        }
        if ( (unsigned int)sub_8D2600(v17) )
        {
LABEL_34:
          v6 = (__int64)v25;
          goto LABEL_35;
        }
        v7 = v17;
        sub_6E5F60(v6 + 28, v17, 8);
        v6 = (__int64)v25;
        goto LABEL_35;
      }
      if ( v25 )
        *v18 = v16;
      else
        v25 = (_QWORD *)v16;
      v6 = *(_QWORD *)(v6 + 16);
    }
    while ( v6 );
    *(_DWORD *)(qword_4D03C50 + 18LL) |= 0x10080u;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) &= ~8u;
    v6 = (__int64)v25;
    v8 = sub_8D3B80(a2);
    if ( v8 )
      goto LABEL_49;
    if ( !*v25 )
    {
      v7 = a2;
      sub_839D30((_DWORD)v25, a2, 1, 0, 0, 1, 0, 0, 0, (__int64)v28, 0, 0);
      goto LABEL_39;
    }
  }
  else
  {
    *(_DWORD *)(qword_4D03C50 + 18LL) |= 0x10080u;
    *(_BYTE *)(v15 + 7) &= ~8u;
    if ( (unsigned int)sub_8D3B80(a2) )
    {
LABEL_49:
      v7 = 0;
      v23 = unk_4D0435C;
      unk_4D0435C = 0;
      sub_6CA0E0(0, 0, 1u, v6, a2, (char *)&dword_4F077C8, v28, 0);
      unk_4D0435C = v23;
LABEL_39:
      v8 = 0;
      if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 )
      {
        v20 = v29;
        if ( v29 )
        {
          v21 = v28[0];
          for ( i = *(unsigned __int8 *)(v28[0] + 140LL); (_BYTE)i == 12; i = *(unsigned __int8 *)(v21 + 140) )
            v21 = *(_QWORD *)(v21 + 160);
          v8 = 0;
          if ( (_BYTE)i )
          {
            v8 = 1;
            if ( v29 == 1 )
            {
              if ( a1 == 31 )
              {
                LOBYTE(v20) = v29 - 1;
                v8 = sub_731B40(v30, v7, i, v20, v19) == 0;
              }
              else if ( a1 == 41 )
              {
                v8 = sub_731D00(v30) == 0;
              }
            }
          }
        }
      }
    }
    else
    {
      v8 = sub_8D2FB0(a2) == 0;
    }
  }
LABEL_35:
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) = (8 * v24)
                                                           | *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7)
                                                           & 0xF7;
LABEL_8:
  sub_6E1990(v6);
  sub_6E2B30(v6, v7);
  sub_6E1DF0(v26);
  return v8;
}
