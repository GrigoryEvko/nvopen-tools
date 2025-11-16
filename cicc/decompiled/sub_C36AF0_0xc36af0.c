// Function: sub_C36AF0
// Address: 0xc36af0
//
__int64 __fastcall sub_C36AF0(__int64 a1, char a2)
{
  char v3; // dl
  unsigned __int8 v4; // al
  unsigned int v5; // r13d
  int v7; // eax
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rdx
  _DWORD *v11; // rax
  __int64 v12; // r13
  unsigned int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // r13
  unsigned int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // r13
  unsigned int v19; // eax
  __int64 v20; // rdi
  unsigned int v21; // r13d
  __int64 v22; // rax
  char v23; // al
  __int64 v24; // rdx
  char v25; // al

  if ( a2 )
  {
    sub_C34440((unsigned __int8 *)a1);
    v3 = *(_BYTE *)(a1 + 20);
    v4 = v3 & 7;
    if ( (v3 & 7) != 2 )
    {
      if ( v4 <= 2u )
        goto LABEL_4;
      v5 = 0;
      if ( v4 != 3 )
      {
        sub_C34440((unsigned __int8 *)a1);
        return v5;
      }
      goto LABEL_10;
    }
LABEL_20:
    if ( sub_C34230(a1) && (*(_BYTE *)(a1 + 20) & 8) != 0 )
    {
      v21 = sub_C337D0(a1);
      v22 = sub_C33900(a1);
      sub_C45D00(v22, 0, v21);
      v23 = *(_BYTE *)(a1 + 20);
      v24 = *(_QWORD *)a1;
      *(_DWORD *)(a1 + 16) = 0;
      v25 = v23 & 0xF8 | 3;
      *(_BYTE *)(a1 + 20) = v25;
      if ( *(_DWORD *)(v24 + 20) == 2 )
        *(_BYTE *)(a1 + 20) = v25 & 0xF7;
      if ( !*(_BYTE *)(v24 + 24) )
      {
        v5 = 0;
        sub_C35A40(a1, 0);
        goto LABEL_11;
      }
      goto LABEL_6;
    }
    if ( (unsigned __int8)sub_C33B00(a1) )
    {
      if ( (*(_BYTE *)(a1 + 20) & 8) == 0 )
      {
        v7 = *(_DWORD *)(*(_QWORD *)a1 + 16LL);
        if ( v7 == 1 )
        {
          sub_C36070(a1, 0, 0, 0);
          v5 = 0;
          goto LABEL_11;
        }
        if ( v7 != 2 )
        {
          v8 = sub_C337D0(a1);
          v9 = sub_C33900(a1);
          v10 = v8;
          v5 = 0;
          sub_C45D00(v9, 0, v10);
          v11 = *(_DWORD **)a1;
          *(_BYTE *)(a1 + 20) &= 0xF8u;
          *(_DWORD *)(a1 + 16) = *v11 + 1;
          goto LABEL_11;
        }
        goto LABEL_6;
      }
    }
    else if ( (*(_BYTE *)(a1 + 20) & 8) == 0 )
    {
      if ( *(void **)a1 == sub_C333E0() || !sub_C33940(a1) && (unsigned __int8)sub_C339A0(a1) )
      {
        v15 = sub_C33900(a1);
        v16 = sub_C337D0(a1);
        sub_C45D00(v15, 0, v16);
        v17 = v15;
        v5 = 0;
        sub_C45DB0(v17, (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 8LL) - 1));
        ++*(_DWORD *)(a1 + 16);
        goto LABEL_11;
      }
      sub_C33F10(a1);
      goto LABEL_6;
    }
    if ( *(_DWORD *)(a1 + 16) == *(_DWORD *)(*(_QWORD *)a1 + 4LL) || !sub_C33BA0(a1) )
    {
      v12 = sub_C33900(a1);
      v13 = sub_C337D0(a1);
      v14 = v12;
      v5 = 0;
      sub_C46E50(v14, 1, v13);
    }
    else
    {
      v18 = sub_C33900(a1);
      v19 = sub_C337D0(a1);
      sub_C46E50(v18, 1, v19);
      v20 = v18;
      v5 = 0;
      sub_C45DB0(v20, (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 8LL) - 1));
      --*(_DWORD *)(a1 + 16);
    }
    goto LABEL_11;
  }
  v3 = *(_BYTE *)(a1 + 20);
  v4 = v3 & 7;
  if ( (v3 & 7) == 2 )
    goto LABEL_20;
  if ( v4 <= 2u )
  {
LABEL_4:
    if ( v4 )
    {
      if ( !(unsigned __int8)sub_C35FD0((_BYTE *)a1) )
        goto LABEL_6;
      v5 = 1;
      sub_C36070(a1, 0, (*(_BYTE *)(a1 + 20) & 8) != 0, 0);
LABEL_11:
      if ( a2 )
        goto LABEL_7;
      return v5;
    }
    if ( (v3 & 8) != 0 )
    {
      v5 = 0;
      sub_C35910(a1, 1);
      goto LABEL_11;
    }
LABEL_6:
    v5 = 0;
    if ( a2 )
    {
LABEL_7:
      sub_C34440((unsigned __int8 *)a1);
      return v5;
    }
    return v5;
  }
  if ( v4 == 3 )
  {
LABEL_10:
    v5 = 0;
    sub_C359D0(a1, 0);
    goto LABEL_11;
  }
  return 0;
}
