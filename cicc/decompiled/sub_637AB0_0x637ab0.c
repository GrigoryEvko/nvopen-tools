// Function: sub_637AB0
// Address: 0x637ab0
//
__int64 __fastcall sub_637AB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r8
  int v11; // eax
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 result; // rax
  char v15; // dl
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rsi
  _OWORD *v19; // rdi
  __int64 v20; // rdx
  __int64 *v21; // r15
  __int64 i; // rax
  __int64 v23; // rax
  unsigned int v24; // eax
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // [rsp-10h] [rbp-260h]
  __int64 v31; // [rsp-8h] [rbp-258h]
  int v32; // [rsp+10h] [rbp-240h]
  int v33; // [rsp+14h] [rbp-23Ch]
  int v34; // [rsp+18h] [rbp-238h]
  __int64 v35; // [rsp+18h] [rbp-238h]
  __int64 *v36; // [rsp+18h] [rbp-238h]
  __int64 v37; // [rsp+18h] [rbp-238h]
  __int64 v38; // [rsp+28h] [rbp-228h] BYREF
  _QWORD v39[2]; // [rsp+30h] [rbp-220h] BYREF
  _OWORD v40[33]; // [rsp+40h] [rbp-210h] BYREF

  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  v38 = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(a1, a2, a3, a4);
  if ( a4 )
  {
    v34 = sub_8D23E0(a4);
    if ( !(unsigned int)sub_8D3A70(a3) || word_4F06418[0] != 28 || v34 )
      goto LABEL_4;
  }
  else
  {
    v34 = sub_8DD3B0(a3);
    if ( !(unsigned int)sub_8D3A70(a3) )
    {
      if ( !v34 )
        goto LABEL_4;
      v15 = *(_BYTE *)(a3 + 140);
      v16 = 0;
      goto LABEL_26;
    }
  }
  v15 = *(_BYTE *)(a3 + 140);
  v23 = a3;
  if ( v15 == 12 )
  {
    do
      v23 = *(_QWORD *)(v23 + 160);
    while ( *(_BYTE *)(v23 + 140) == 12 );
  }
  v16 = *(_QWORD *)(*(_QWORD *)v23 + 96LL);
  if ( !v16 )
  {
    if ( !v34 )
      goto LABEL_4;
    goto LABEL_26;
  }
  if ( *(_QWORD *)(v16 + 8) )
  {
    memset(v40, 0, 48);
    if ( !dword_4F077BC || qword_4F077A8 > 0x9F5Fu )
    {
LABEL_33:
      DWORD2(v40[2]) |= 0x8000009u;
      if ( !v34 )
      {
        if ( *(_BYTE *)(a2 + 8) == 2 )
          v8 = a3;
        else
          BYTE11(v40[2]) |= 0x10u;
        v18 = v8;
        v19 = (_OWORD *)a3;
        sub_6C64D0(a3, v8, (unsigned int)&v38, dword_4D048B8, 0, 0, (__int64)v40);
        result = v30;
        v20 = v31;
LABEL_37:
        v12 = *((_QWORD *)&v40[0] + 1);
        if ( *((_QWORD *)&v40[0] + 1) )
        {
          if ( (*(_BYTE *)(a1 + 193) & 2) != 0 )
            result = sub_630970(a1, (__int64)v40, (__int64)&v38);
        }
        else
        {
          result = sub_72C9D0(v19, v18, v20);
          v12 = result;
        }
        goto LABEL_20;
      }
LABEL_88:
      v18 = 0;
      v19 = v40;
      result = sub_6C56C0(v40, 0);
      goto LABEL_37;
    }
LABEL_32:
    BYTE10(v40[2]) |= 1u;
    goto LABEL_33;
  }
  if ( v34 )
  {
LABEL_26:
    if ( v15 == 12 )
    {
      v17 = a3;
      do
      {
        v17 = *(_QWORD *)(v17 + 160);
        v15 = *(_BYTE *)(v17 + 140);
      }
      while ( v15 == 12 );
    }
    if ( v15 )
    {
      memset(v40, 0, 48);
      if ( !dword_4F077BC || qword_4F077A8 > 0x9F5Fu )
      {
        DWORD2(v40[2]) |= 0x8000009u;
        goto LABEL_88;
      }
      goto LABEL_32;
    }
  }
  if ( word_4F06418[0] != 28 || !v16 )
  {
    v34 = 0;
    goto LABEL_4;
  }
  v34 = sub_876D90(a3, v8, dword_4F07508, 1, 0);
  if ( !v34 )
  {
LABEL_4:
    sub_6BB6B0(v39);
    if ( dword_4F04C64 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) != 0 )
    {
      v21 = (__int64 *)v39[0];
      if ( v39[0] )
      {
        v32 = 0;
        v33 = 0;
        if ( (unsigned int)sub_6E1B40(v39[0]) )
          goto LABEL_46;
LABEL_42:
        if ( v33 != 1 )
        {
          v33 = 1;
          for ( i = *v21; *v21; i = *v21 )
          {
            if ( *(_BYTE *)(i + 8) == 3 )
            {
              i = sub_6BBB10(v21);
              if ( !i )
                break;
            }
            v21 = (__int64 *)i;
            if ( !(unsigned int)sub_6E1B40(i) )
              goto LABEL_42;
LABEL_46:
            v32 = 1;
          }
          if ( v32 )
          {
            memset(v40, 0, 48);
            if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
              BYTE10(v40[2]) |= 1u;
            DWORD2(v40[2]) |= 0x8000009u;
            sub_6C56C0(v40, v39[0]);
            sub_6E1BF0(v39);
            v12 = *((_QWORD *)&v40[0] + 1);
            goto LABEL_19;
          }
        }
      }
    }
    v9 = sub_6E1C80(v39);
    v10 = v9;
    if ( !v9 )
    {
      v24 = sub_8D32E0(a3);
      v25 = v24;
      if ( v24 )
      {
        sub_6851C0(808, dword_4F07508);
        v12 = sub_72C9D0(808, dword_4F07508, v29);
      }
      else
      {
        if ( !dword_4F077BC || !unk_4D04228 || !a2 || *(_BYTE *)(a2 + 8) == 2 )
          v25 = v34 == 0;
        v12 = sub_725A70(v25);
      }
      goto LABEL_19;
    }
    if ( v34 )
    {
      v37 = v9;
      sub_6851C0(1049, &dword_4F063F8);
      v10 = v37;
    }
    if ( dword_4D04428 && dword_4F077BC )
    {
      if ( !a4 )
      {
        memset(v40, 0, 0x1D8u);
        *((_QWORD *)&v40[9] + 1) = v40;
        *((_QWORD *)&v40[1] + 1) = *(_QWORD *)&dword_4F063F8;
        goto LABEL_61;
      }
      if ( *(_BYTE *)(v10 + 8) == 1 )
      {
        v36 = (__int64 *)v10;
        sub_684B30(2365, &dword_4F063F8);
        sub_637960(a1, a4, a2, v36);
        sub_6E1990(v36);
        v12 = *(_QWORD *)(a2 + 24);
        goto LABEL_17;
      }
    }
    else if ( !a4 )
    {
      goto LABEL_13;
    }
    v35 = v10;
    v11 = sub_8D3770(a4);
    v10 = v35;
    if ( v11 )
    {
LABEL_13:
      memset(v40, 0, 0x1D8u);
      *((_QWORD *)&v40[9] + 1) = v40;
      *((_QWORD *)&v40[1] + 1) = *(_QWORD *)&dword_4F063F8;
      if ( !dword_4F077BC )
      {
LABEL_14:
        LOBYTE(v40[11]) |= 8u;
        *(_QWORD *)&v40[18] = a3;
        sub_6E1C20(v10, 1, (char *)&v40[20] + 8);
        sub_632300((__int64 *)v40, 0, 0, (__int64)&v38);
        if ( (*(_BYTE *)(a1 + 193) & 2) != 0 )
          sub_630970(a1, (__int64)&v40[8] + 8, (__int64)&v38);
        v12 = *(_QWORD *)&v40[9];
        goto LABEL_17;
      }
LABEL_61:
      if ( qword_4F077A8 <= 0x9F5Fu )
        BYTE2(v40[11]) |= 1u;
      goto LABEL_14;
    }
    v12 = sub_6D70D0(a2, v35);
    if ( !v12 )
    {
      v12 = sub_72C9D0(a2, v35, v28);
      if ( !v39[0] )
      {
LABEL_19:
        unk_4F061D8 = *(_QWORD *)&dword_4F063F8;
        result = sub_7BE280(28, 18, 0, 0);
        goto LABEL_20;
      }
      sub_6E1BF0(v39);
    }
LABEL_17:
    if ( v39[0] )
    {
      v13 = sub_6E1A20(v39[0]);
      sub_6851C0(146, v13);
      sub_6E1BF0(v39);
    }
    goto LABEL_19;
  }
  v26 = dword_4D04230;
  if ( dword_4D04230 )
  {
    v26 = 1;
    if ( dword_4F077BC )
      v26 = unk_4D04228 == 0;
  }
  v12 = sub_725A70(v26);
  unk_4F061D8 = *(_QWORD *)&dword_4F063F8;
  result = sub_7B8B50(v26, v8, *(_QWORD *)&dword_4F063F8, v27);
LABEL_20:
  *(_BYTE *)(v12 + 49) |= 0x20u;
  if ( a2 )
    *(_QWORD *)(a2 + 24) = v12;
  return result;
}
