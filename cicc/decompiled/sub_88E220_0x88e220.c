// Function: sub_88E220
// Address: 0x88e220
//
__int64 __fastcall sub_88E220(__int64 a1)
{
  __int64 v1; // r12
  unsigned int v2; // ebx
  unsigned int v3; // r15d
  unsigned int v4; // r14d
  unsigned int v5; // r13d
  __int64 v6; // rax
  __int64 v7; // rax
  char v8; // al
  __int64 i; // rax
  char v10; // al
  __int64 v12; // rax
  unsigned int v13; // edi
  char *v14; // rcx
  __int64 v15; // [rsp+8h] [rbp-458h]
  unsigned int v16; // [rsp+18h] [rbp-448h] BYREF
  int v17; // [rsp+1Ch] [rbp-444h] BYREF
  char *v18; // [rsp+20h] [rbp-440h] BYREF
  __int64 v19; // [rsp+28h] [rbp-438h] BYREF
  char s[1072]; // [rsp+30h] [rbp-430h] BYREF

  v1 = a1;
  v2 = dword_4F60184 == 0 ? 3641 : 3586;
  v3 = dword_4F60184 == 0 ? 3639 : 3585;
  v4 = dword_4F60184 == 0 ? 3638 : 3583;
  v5 = dword_4F60184 == 0 ? 3637 : 3582;
  if ( (*(_BYTE *)(a1 + 89) & 1) == 0 )
  {
LABEL_7:
    if ( !(unsigned int)sub_8D3AA0(a1) )
      goto LABEL_9;
    v7 = *(_QWORD *)(v1 + 168);
    if ( (*(_BYTE *)(v7 + 109) & 0x20) == 0 )
      goto LABEL_9;
    if ( *(_QWORD *)(v7 + 96) )
      goto LABEL_26;
LABEL_33:
    v13 = *(_DWORD *)(v1 + 64);
    v14 = (char *)byte_3F871B3;
    if ( v13 )
    {
      sub_729E00(v13, &v18, &v19, &v16, &v17);
      v14 = (char *)byte_3F871B3;
      if ( !v17 )
      {
        if ( v16 )
        {
          v18 = (char *)sub_723260(v18);
          snprintf(s, 0x400u, ", defined at %s:%lu", v18, v16);
          v14 = s;
        }
      }
    }
    sub_686310(7u, v5, dword_4F07508, (__int64)v14, v1);
    v8 = *(_BYTE *)(v1 + 89);
    goto LABEL_27;
  }
  do
  {
    v6 = sub_72B7D0(a1);
    if ( !v6 )
    {
      a1 = v1;
      goto LABEL_7;
    }
    if ( (*(_BYTE *)(v6 + 89) & 5) != 5 )
      break;
    a1 = *(_QWORD *)(*(_QWORD *)(v6 + 40) + 32LL);
  }
  while ( (*(_BYTE *)(a1 + 89) & 1) != 0 );
  v15 = v6;
  if ( (unsigned int)sub_8D3AA0(v1) && (v12 = *(_QWORD *)(v1 + 168), (*(_BYTE *)(v12 + 109) & 0x20) != 0) )
  {
    if ( *(_QWORD *)(v12 + 96) )
    {
      v10 = *(_BYTE *)(v15 + 198);
      if ( (v10 & 0x10) == 0 || (v10 & 0x18) == 0x18 )
        goto LABEL_26;
    }
    else
    {
      v10 = *(_BYTE *)(v15 + 198);
      if ( (v10 & 0x10) == 0 )
        goto LABEL_33;
    }
  }
  else
  {
    v10 = *(_BYTE *)(v15 + 198);
    if ( (v10 & 0x10) == 0 )
    {
LABEL_9:
      v8 = *(_BYTE *)(v1 + 89);
      if ( (v8 & 1) == 0 )
      {
        if ( *(_QWORD *)(v1 + 8) )
        {
LABEL_27:
          if ( (v8 & 4) != 0 && (unsigned __int8)((*(_BYTE *)(v1 + 88) & 3) - 1) <= 1u )
            sub_685260(7u, v3, dword_4F07508, v1);
          return 0;
        }
        goto LABEL_11;
      }
      sub_685260(7u, v4, dword_4F07508, v1);
      if ( !*(_QWORD *)(v1 + 8) )
      {
LABEL_11:
        if ( (unsigned int)sub_8D2870(v1) )
          goto LABEL_16;
        if ( (unsigned int)sub_8D3A70(v1) )
        {
          for ( i = v1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          if ( (*(_BYTE *)(i + 177) & 4) != 0 )
LABEL_16:
            sub_685260(7u, v2, dword_4F07508, v1);
        }
      }
LABEL_26:
      v8 = *(_BYTE *)(v1 + 89);
      goto LABEL_27;
    }
  }
  dword_4F60180 = 1;
  if ( unk_4F60218 && (v10 & 8) != 0 )
    sub_686EE0(5u, 0xE82u, (_DWORD *)(v1 + 64), *(_QWORD *)(v15 + 8), *(_QWORD *)(unk_4F60218 + 8LL), v1);
  return 0;
}
