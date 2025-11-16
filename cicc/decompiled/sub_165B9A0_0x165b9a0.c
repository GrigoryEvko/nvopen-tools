// Function: sub_165B9A0
// Address: 0x165b9a0
//
void __fastcall sub_165B9A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r15
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // r15
  _BYTE *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rax
  _QWORD v17[2]; // [rsp+0h] [rbp-50h] BYREF
  char v18; // [rsp+10h] [rbp-40h]
  char v19; // [rsp+11h] [rbp-3Fh]

  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 )
    return;
  v3 = 0x2000006000001LL;
  while ( 1 )
  {
    v6 = (__int64)sub_1648700(v2);
    v7 = *(unsigned __int8 *)(v6 + 16);
    if ( (unsigned __int8)v7 <= 0x17u || (unsigned __int8)(v7 - 29) > 0x31u || !_bittest64(&v3, (unsigned int)(v7 - 29)) )
      break;
    switch ( (_BYTE)v7 )
    {
      case 0x37:
        v16 = *(_QWORD *)(v6 - 24);
        if ( !v16 || v16 != a2 )
        {
          v19 = 1;
          v17[0] = "swifterror value should be the second operand when used by stores";
          v18 = 3;
          sub_164FF40((__int64 *)a1, (__int64)v17);
          if ( *(_QWORD *)a1 )
          {
            sub_164FA80((__int64 *)a1, a2);
            sub_164FA80((__int64 *)a1, v6);
          }
          return;
        }
        break;
      case 0x4E:
        sub_165B7E0((__int64 *)a1, v6 | 4, a2);
        if ( *(_BYTE *)(v6 + 16) == 29 )
LABEL_27:
          sub_165B7E0((__int64 *)a1, v6 & 0xFFFFFFFFFFFFFFFBLL, a2);
        break;
      case 0x1D:
        goto LABEL_27;
    }
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return;
  }
  v8 = *(_QWORD *)a1;
  v19 = 1;
  v17[0] = "swifterror value can only be loaded and stored from, or as a swifterror argument!";
  v18 = 3;
  if ( !v8 )
  {
    *(_BYTE *)(a1 + 72) = 1;
    return;
  }
  sub_16E2CE0(v17, v8);
  v9 = *(_BYTE **)(v8 + 24);
  if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 16) )
  {
    sub_16E7DE0(v8, 10);
  }
  else
  {
    *(_QWORD *)(v8 + 24) = v9 + 1;
    *v9 = 10;
  }
  v10 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 72) = 1;
  if ( v10 )
  {
    if ( *(_BYTE *)(a2 + 16) <= 0x17u )
    {
      sub_1553920((__int64 *)a2, v10, 1, a1 + 16);
      v11 = *(_QWORD *)a1;
      v12 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v12 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        goto LABEL_25;
LABEL_17:
      *(_QWORD *)(v11 + 24) = v12 + 1;
      *v12 = 10;
    }
    else
    {
      sub_155BD40(a2, v10, a1 + 16, 0);
      v11 = *(_QWORD *)a1;
      v12 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v12 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        goto LABEL_17;
LABEL_25:
      sub_16E7DE0(v11, 10);
    }
    v13 = *(_QWORD *)a1;
    if ( *(_BYTE *)(v6 + 16) <= 0x17u )
    {
      sub_1553920((__int64 *)v6, v13, 1, a1 + 16);
      v14 = *(_QWORD *)a1;
      v15 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v15 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        goto LABEL_20;
    }
    else
    {
      sub_155BD40(v6, v13, a1 + 16, 0);
      v14 = *(_QWORD *)a1;
      v15 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v15 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
LABEL_20:
        *(_QWORD *)(v14 + 24) = v15 + 1;
        *v15 = 10;
        return;
      }
    }
    sub_16E7DE0(v14, 10);
  }
}
