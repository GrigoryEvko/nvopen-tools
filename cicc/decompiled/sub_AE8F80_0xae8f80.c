// Function: sub_AE8F80
// Address: 0xae8f80
//
void __fastcall sub_AE8F80(char *a1)
{
  __int64 v1; // rsi
  char v2; // al
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // r12d
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // rsi
  _QWORD v16[6]; // [rsp-30h] [rbp-30h] BYREF

  v1 = *((_QWORD *)a1 + 6);
  if ( !v1 )
    return;
  v2 = *a1;
  if ( *a1 == 85 )
  {
    v3 = *((_QWORD *)a1 - 4);
    if ( v3
      && !*(_BYTE *)v3
      && *(_QWORD *)(v3 + 24) == *((_QWORD *)a1 + 10)
      && (*(_BYTE *)(v3 + 33) & 0x20) != 0
      && !(unsigned __int8)sub_B58D90(*(unsigned int *)(v3 + 36)) )
    {
      v15 = *((_QWORD *)a1 + 6);
      v16[0] = 0;
      if ( v15 )
        goto LABEL_17;
      return;
    }
  }
  else if ( v2 != 34 && v2 != 40 )
  {
    v16[0] = 0;
    goto LABEL_17;
  }
  v4 = sub_B43CB0(a1);
  v5 = sub_B92180(v4);
  v7 = v5;
  if ( v5 )
  {
    v8 = sub_BD5C60(a1, v1, v6);
    v9 = sub_B01860(v8, 0, 0, v7, 0, 0, 0, 1);
    sub_B10CB0(v16, v9);
    if ( *((_QWORD *)a1 + 6) )
      sub_B91220(a1 + 48);
    v13 = v16[0];
    *((_QWORD *)a1 + 6) = v16[0];
    if ( v13 )
      sub_B976B0(v16, v13, a1 + 48, v10, v11, v12);
  }
  else
  {
    v14 = *((_QWORD *)a1 + 6);
    v16[0] = 0;
    if ( v14 )
    {
LABEL_17:
      sub_B91220(a1 + 48);
      *((_QWORD *)a1 + 6) = v16[0];
    }
  }
}
