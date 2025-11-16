// Function: sub_626F50
// Address: 0x626f50
//
__int64 __fastcall sub_626F50(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  __int64 v10; // rbx
  __int64 v11; // rax
  bool v12; // zf
  __int64 v13; // rcx
  char *v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  char v19; // al
  __int64 result; // rax
  _BYTE *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v27; // [rsp+8h] [rbp-58h]
  unsigned int v28; // [rsp+8h] [rbp-58h]
  int v29; // [rsp+18h] [rbp-48h] BYREF
  unsigned int v30; // [rsp+1Ch] [rbp-44h] BYREF
  int v31; // [rsp+20h] [rbp-40h] BYREF
  int v32; // [rsp+24h] [rbp-3Ch] BYREF
  _QWORD v33[7]; // [rsp+28h] [rbp-38h] BYREF

  v10 = a1;
  v29 = (a1 >> 5) & 1;
  v33[0] = 0;
  v30 = 0;
  *(_QWORD *)(a2 + 40) = *(_QWORD *)&dword_4F063F8;
  v31 = 0;
  v32 = 0;
  v11 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (dword_4F04C44 != -1 || (*(_BYTE *)(v11 + 6) & 2) != 0) && *(_BYTE *)(v11 + 4) == 6 )
  {
    v21 = *(_BYTE **)(a2 + 288);
    if ( (v21[89] & 4) != 0 )
    {
      if ( *(_QWORD *)(*(_QWORD *)v21 + 72LL) )
      {
        if ( *(char *)(*(_QWORD *)v21 + 81LL) >= 0 )
        {
          v27 = *(_QWORD *)v21;
          v22 = sub_7D2A80(*(_QWORD *)v21);
          if ( v27 != v22 )
          {
            v23 = *(_QWORD *)(v22 + 88);
            *(_QWORD *)(a2 + 288) = v23;
            *(_QWORD *)(a2 + 280) = v23;
            *(_QWORD *)(a2 + 272) = v23;
          }
        }
      }
    }
  }
  if ( a6 )
  {
    a6[6] = *(_QWORD *)&dword_4F063F8;
    a6[7] = qword_4F063F0;
  }
  v12 = *(_BYTE *)(a2 + 268) == 4;
  unk_4F061D8 = qword_4F063F0;
  if ( v12 )
  {
    BYTE1(v10) = BYTE1(a1) | 1;
  }
  else if ( a3 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 8) != 0 )
    {
      BYTE1(v10) = BYTE1(a1) | 4;
      LODWORD(a3) = 0;
    }
    else if ( dword_4F077C4 == 2 && *(_BYTE *)(a2 + 269) != 2 )
    {
      v10 = a1 | 0x10;
    }
  }
  v13 = *(_QWORD *)(a2 + 288);
  v14 = (char *)(a2 + 16);
  v15 = v10;
  sub_62C0A0(
    v10,
    (_DWORD)v14,
    a2,
    v13,
    a3,
    a4,
    a2 + 280,
    (__int64)v33,
    (__int64)&v29,
    (__int64)&v31,
    (__int64)&v30,
    (__int64)&v32,
    0,
    a2 + 352,
    a5,
    (__int64)a6);
  *(_QWORD *)(a2 + 48) = *(_QWORD *)dword_4F07508;
  if ( (*(_BYTE *)(a2 + 129) & 0x40) != 0 && dword_4D043E8 )
  {
    v28 = 0;
    do
    {
      v14 = "override";
      v15 = 240;
      if ( (unsigned int)sub_7C8F50(240, "override") )
      {
        if ( (*(_BYTE *)(a5 + 65) & 4) != 0 )
          goto LABEL_79;
        if ( dword_4F077BC && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
        {
          v15 = 2466;
          sub_684B30(2466, a2 + 48);
        }
        *(_BYTE *)(a5 + 65) |= 4u;
      }
      else
      {
        if ( !dword_4D043E8 )
          break;
        v14 = "final";
        v15 = 241;
        if ( !(unsigned int)sub_7C8F50(241, "final") )
        {
          if ( HIDWORD(qword_4F077B4) )
          {
            if ( !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x9EFBu )
              break;
          }
          else if ( !(_DWORD)qword_4F077B4 )
          {
            break;
          }
          v14 = "__final";
          v15 = 241;
          if ( !(unsigned int)sub_7C8F50(241, "__final") )
            break;
        }
        if ( (*(_BYTE *)(a5 + 65) & 2) != 0 )
        {
LABEL_79:
          v14 = (char *)(a2 + 48);
          v15 = 1534;
          sub_6851C0(1534, a2 + 48);
          v28 = 1;
          goto LABEL_58;
        }
        if ( dword_4F077BC )
        {
          if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && (v15 = dword_4F07774) == 0 )
          {
            v15 = 2466;
            sub_684B30(2466, a2 + 48);
          }
        }
        *(_BYTE *)(a5 + 65) |= 2u;
      }
      v14 = (char *)v28;
      if ( !v28 )
      {
        if ( (*(_BYTE *)(a2 + 130) & 1) != 0 )
        {
          if ( (*(_BYTE *)(a2 + 122) & 4) == 0 )
            goto LABEL_58;
          v15 = 3247;
          if ( (*(_BYTE *)(a2 + 129) & 0x10) != 0 )
            goto LABEL_58;
        }
        else
        {
          v15 = 2884;
        }
        v14 = (char *)&dword_4F063F8;
        sub_6851C0(v15, &dword_4F063F8);
        *(_BYTE *)(a5 + 65) &= 0xF9u;
        v28 = 1;
      }
LABEL_58:
      sub_7B8B50(v15, v14, v25, v16);
      v16 = dword_4D043E8;
    }
    while ( dword_4D043E8 );
  }
  v17 = *(_QWORD *)(a2 + 16);
  if ( v29 )
  {
    v17 |= 0x10uLL;
    *(_QWORD *)(a2 + 16) = v17;
  }
  v18 = v30;
  if ( v30 )
  {
    v17 |= 0x20uLL;
    *(_QWORD *)(a2 + 16) = v17;
  }
  if ( (v17 & 0x80u) != 0LL )
  {
    v15 = *(_QWORD *)(a2 + 280);
    v14 = (char *)(a2 + 40);
    sub_8DD040(v15, a2 + 40);
  }
  if ( word_4F06418[0] == 27 && (v10 & 8) != 0 )
  {
    v24 = *(_QWORD *)(a2 + 16);
    if ( (v24 & 1) == 0 )
    {
      *(_QWORD *)(a2 + 16) = v24 | 1;
      if ( a6 )
        a6[8] = *(_QWORD *)&dword_4F063F8;
      sub_7B8B50(v15, v14, v18, v16);
    }
  }
  v19 = *(_BYTE *)(a2 + 124);
  if ( (v19 & 0x20) != 0 )
  {
    sub_6451E0(a2);
    if ( *(char *)(a2 + 124) >= 0 )
      goto LABEL_19;
LABEL_61:
    sub_625720(a2);
    goto LABEL_21;
  }
  if ( v19 < 0 )
    goto LABEL_61;
LABEL_19:
  if ( a4 && (*(_BYTE *)(a4 + 16) & 0x10) != 0 && unk_4F0774C && (unsigned int)sub_8D3EA0(v33[0]) )
    *(_BYTE *)(a2 + 125) |= 8u;
LABEL_21:
  result = *(_QWORD *)(a2 + 280);
  *(_QWORD *)(a2 + 288) = result;
  if ( !result )
  {
    result = sub_72C930();
    *(_QWORD *)(a2 + 288) = result;
  }
  if ( word_4F06418[0] == 294 && (*(_BYTE *)(a2 + 122) & 0x10) == 0 )
  {
    result = *(_QWORD *)(a2 + 288);
    if ( *(_BYTE *)(result + 140) != 7 || !*(_QWORD *)(a2 + 368) || (*(_BYTE *)(a2 + 133) & 8) != 0 )
      return sub_623350(a2, (unsigned int *)a5, a4);
  }
  return result;
}
