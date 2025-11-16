// Function: sub_65C7C0
// Address: 0x65c7c0
//
__int64 __fastcall sub_65C7C0(__int64 a1)
{
  __int16 v2; // dx
  __int16 v3; // ax
  char v4; // al
  unsigned __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 i; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  char k; // dl
  __int64 v15; // rax
  char j; // dl
  __int64 *v17; // rax
  __int64 **v18; // rcx
  _QWORD *v19; // rdx
  __int64 **v20; // r14
  _QWORD **v21; // rax
  _QWORD *v22; // rsi
  __int64 *v23; // rax
  __int64 **v24; // rcx
  _QWORD *v25; // rdx
  _QWORD *v26; // r13
  __int64 v27; // rax
  _QWORD *v28; // rdi
  _QWORD *m; // rax
  __int64 *v30; // r13
  __int64 *v31; // r15
  _QWORD *v32; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v33[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(_WORD *)(a1 + 122) & 0xF7DF;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  v3 = v2 | ((word_4D04430 & 1) << 11) | 0x20;
  *(_WORD *)(a1 + 122) = v3;
  if ( (v3 & 0x100) == 0 )
  {
    v4 = *(_BYTE *)(a1 + 132);
    if ( (v4 & 0x40) == 0 )
      *(_BYTE *)(a1 + 132) = v4 | 0x80;
  }
  *(_QWORD *)(a1 + 24) = *(_QWORD *)&dword_4F063F8;
  v5 = (-(__int64)((*(_BYTE *)(a1 + 122) & 0x10) == 0) & 0xFFFFFFFFFF000000LL) + 17301506;
  v6 = a1;
  if ( dword_4D043F8 )
    v5 = ((-(__int64)((*(_BYTE *)(a1 + 122) & 0x10) == 0) & 0xFFFFFFFFFF000000LL) + 17301506) | 0x8000000;
  if ( dword_4D043E0 )
    v5 |= 0x400000u;
  sub_672A20(v5, a1, 0);
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
  {
    v6 = *(_QWORD *)(a1 + 272);
    v5 = a1 + 24;
    sub_64E990(a1 + 24, v6, 0, 0, 0, 1);
  }
  for ( i = *(_QWORD *)(a1 + 288); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  *(_BYTE *)(i + 88) |= 4u;
  v10 = word_4F06418[0] & 0xFFFD;
  if ( (word_4F06418[0] & 0xFFFD) == 0x19 || word_4F06418[0] == 34 )
    goto LABEL_27;
  if ( dword_4F077C4 != 2 )
    goto LABEL_18;
  if ( word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0 )
  {
    if ( (v6 = 0, v5 = 0, !(unsigned int)sub_7C0F00(0, 0)) && word_4F06418[0] == 15
      || word_4F06418[0] == 33
      || (v10 = dword_4D04474) != 0 && word_4F06418[0] == 52 )
    {
LABEL_27:
      if ( !unk_4D047EC || dword_4F04C58 == -1 || (*(_BYTE *)(a1 + 124) & 4) != 0 )
      {
        v6 = a1;
        v5 = 6;
        sub_626F50(6u, a1, 0, 0, 0, 0);
      }
      else
      {
        v6 = a1;
        sub_626F50(0xC006u, a1, 0, 0, 0, 0);
        v5 = *(_QWORD *)(a1 + 288);
        if ( (unsigned int)sub_8D3410(v5) )
        {
          v5 = *(_QWORD *)(a1 + 288);
          if ( (unsigned int)sub_8DCFE0(v5) )
          {
            v6 = a1 + 24;
            v5 = 890;
            sub_6851C0(890, a1 + 24);
          }
        }
      }
    }
  }
  if ( dword_4F077C4 == 2 )
  {
    v5 = a1;
    sub_65C470(a1, v6, v10, v7, v8);
  }
LABEL_18:
  if ( qword_4D0495C )
  {
    v5 = *(_QWORD *)(a1 + 288);
    if ( (unsigned int)sub_6454D0(v5, a1 + 24) )
      goto LABEL_20;
  }
  v15 = *(_QWORD *)(a1 + 288);
  for ( j = *(_BYTE *)(v15 + 140); j == 12; j = *(_BYTE *)(v15 + 140) )
    v15 = *(_QWORD *)(v15 + 160);
  if ( j == 21 )
  {
LABEL_20:
    v11 = sub_72C930(v5);
    *(_QWORD *)(a1 + 272) = v11;
    *(_QWORD *)(a1 + 280) = v11;
    *(_QWORD *)(a1 + 288) = v11;
    if ( (*(_BYTE *)(a1 + 124) & 0x20) != 0 )
      goto LABEL_44;
  }
  else if ( (*(_BYTE *)(a1 + 124) & 0x20) != 0 )
  {
LABEL_44:
    sub_6451E0(a1);
    if ( *(_QWORD *)(a1 + 184) )
      goto LABEL_22;
    goto LABEL_45;
  }
  if ( *(_QWORD *)(a1 + 184) )
    goto LABEL_22;
LABEL_45:
  if ( !*(_QWORD *)(a1 + 200) )
    goto LABEL_25;
LABEL_22:
  v12 = *(_QWORD *)(a1 + 288);
  for ( k = *(_BYTE *)(v12 + 140); k == 12; k = *(_BYTE *)(v12 + 140) )
    v12 = *(_QWORD *)(v12 + 160);
  if ( k )
  {
    v17 = *(__int64 **)(a1 + 200);
    v32 = 0;
    v18 = (__int64 **)(a1 + 200);
    v33[0] = 0;
    if ( v17 )
    {
      v19 = v33;
      do
      {
        while ( *((_BYTE *)v17 + 9) != 2 && (*((_BYTE *)v17 + 11) & 0x10) == 0 )
        {
          v18 = (__int64 **)v17;
          v17 = (__int64 *)*v17;
          if ( !v17 )
            goto LABEL_53;
        }
        *v19 = v17;
        *v18 = (__int64 *)*v17;
        v19 = (_QWORD *)*v19;
        *v19 = 0;
        v17 = *v18;
      }
      while ( *v18 );
LABEL_53:
      v17 = (__int64 *)v33[0];
    }
    v32 = v17;
    v20 = (__int64 **)(a1 + 184);
    v21 = sub_5CB9F0(&v32);
    v33[0] = 0;
    v22 = v21;
    v23 = *(__int64 **)(a1 + 184);
    if ( v23 )
    {
      v24 = (__int64 **)(a1 + 184);
      v25 = v33;
      do
      {
        while ( *((_BYTE *)v23 + 9) != 2 && (*((_BYTE *)v23 + 11) & 0x10) == 0 )
        {
          v24 = (__int64 **)v23;
          v23 = (__int64 *)*v23;
          if ( !v23 )
            goto LABEL_60;
        }
        *v25 = v23;
        *v24 = (__int64 *)*v23;
        v25 = (_QWORD *)*v25;
        *v25 = 0;
        v23 = *v24;
      }
      while ( *v24 );
LABEL_60:
      v23 = (__int64 *)v33[0];
    }
    *v22 = v23;
    v26 = v32;
    if ( v32 )
    {
      do
      {
        v26[6] = a1;
        if ( (unsigned int)sub_8D3A70(*(_QWORD *)(a1 + 288)) || (unsigned int)sub_8D2870(*(_QWORD *)(a1 + 288)) )
          sub_5CCAE0(5u, (__int64)v26);
        v26 = (_QWORD *)*v26;
      }
      while ( v26 );
      v27 = sub_5CEF40(*(_QWORD *)(a1 + 288), 0);
      v28 = v32;
      *(_QWORD *)(a1 + 288) = v27;
      sub_5CEC90(v28, v27, 6);
      for ( m = v32; m; m = (_QWORD *)*m )
        m[6] = 0;
    }
    v30 = *(__int64 **)(a1 + 184);
    if ( !v30 )
      goto LABEL_75;
    if ( *(_QWORD *)(a1 + 192) )
    {
      do
      {
        while ( 1 )
        {
          v31 = v30;
          v30 = (__int64 *)*v30;
          if ( *((_BYTE *)v31 + 9) == 4 )
            break;
          v20 = (__int64 **)v31;
          if ( !v30 )
            goto LABEL_74;
        }
        *v31 = 0;
        *sub_5CB9F0(*(_QWORD ***)(a1 + 192)) = v31;
        *v20 = v30;
      }
      while ( v30 );
LABEL_74:
      v30 = *(__int64 **)(a1 + 184);
      if ( !v30 )
      {
LABEL_75:
        if ( !*(_QWORD *)(a1 + 200) )
          goto LABEL_25;
        v30 = 0;
      }
    }
    sub_644730(v30);
    sub_644730(*(__int64 **)(a1 + 200));
  }
LABEL_25:
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 24);
  return sub_643EB0(a1, 0);
}
