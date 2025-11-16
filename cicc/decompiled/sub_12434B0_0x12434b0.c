// Function: sub_12434B0
// Address: 0x12434b0
//
__int64 __fastcall sub_12434B0(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  __int64 result; // rax
  unsigned __int64 v4; // r14
  int v5; // eax
  int v6; // eax
  unsigned __int64 v7; // rsi
  char v8; // al
  unsigned __int64 v9; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v10; // [rsp+0h] [rbp-A0h]
  char v11; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v12; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v13; // [rsp+0h] [rbp-A0h]
  unsigned __int8 v14; // [rsp+8h] [rbp-98h]
  unsigned __int8 v15; // [rsp+8h] [rbp-98h]
  unsigned __int8 v16; // [rsp+8h] [rbp-98h]
  unsigned __int64 v17; // [rsp+10h] [rbp-90h] BYREF
  __int64 v18; // [rsp+18h] [rbp-88h] BYREF
  _BYTE *v19; // [rsp+20h] [rbp-80h] BYREF
  __int64 v20; // [rsp+28h] [rbp-78h]
  _QWORD v21[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v22[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v23[2]; // [rsp+50h] [rbp-50h] BYREF
  char v24; // [rsp+60h] [rbp-40h]
  char v25; // [rsp+61h] [rbp-3Fh]

  v2 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  LOBYTE(v21[0]) = 0;
  v4 = *(_QWORD *)(a1 + 232);
  v5 = *(_DWORD *)(a1 + 240);
  v19 = v21;
  v20 = 0;
  v17 = 0;
  if ( v5 == 412 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(v2);
    if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") || (unsigned __int8)sub_120C050(a1, (__int64 *)&v17) )
      goto LABEL_8;
  }
  else
  {
    if ( v5 != 413 )
    {
      v25 = 1;
      v22[0] = (__int64)"expected name or guid tag";
      v24 = 3;
      sub_11FD800(v2, v4, (__int64)v22, 1);
      goto LABEL_8;
    }
    *(_DWORD *)(a1 + 240) = sub_1205200(v2);
    if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") || (unsigned __int8)sub_120B3D0(a1, (__int64)&v19) )
      goto LABEL_8;
  }
  if ( *(_DWORD *)(a1 + 240) != 4 )
  {
    if ( !(unsigned __int8)sub_120AFE0(a1, 13, "expected ')' here") )
    {
      v22[0] = (__int64)v23;
      v9 = v17;
      v18 = 0;
      sub_12060D0(v22, v19, (__int64)&v19[v20]);
      result = sub_123DE00(a1, v22, v9, 0, a2, &v18, v4);
      if ( (_QWORD *)v22[0] != v23 )
      {
        v15 = result;
        j_j___libc_free_0(v22[0], v23[0] + 1LL);
        result = v15;
      }
      if ( v18 )
      {
        v16 = result;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
        result = v16;
      }
      goto LABEL_9;
    }
    goto LABEL_8;
  }
  *(_DWORD *)(a1 + 240) = sub_1205200(v2);
  if ( (unsigned __int8)sub_120AFE0(a1, 414, "expected 'summaries' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    goto LABEL_8;
  }
  do
  {
    v6 = *(_DWORD *)(a1 + 240);
    switch ( v6 )
    {
      case 426:
        v10 = v17;
        sub_2241BD0(v22, &v19);
        v8 = sub_1242650(a1, (__int64)v22, v10, a2);
        break;
      case 447:
        v13 = v17;
        sub_2241BD0(v22, &v19);
        v8 = sub_123E5A0(a1, (__int64)v22, v13, a2);
        break;
      case 98:
        v12 = v17;
        sub_2241BD0(v22, &v19);
        v8 = sub_123EA90(a1, (__int64)v22, v12, a2);
        break;
      default:
        v25 = 1;
        v7 = *(_QWORD *)(a1 + 232);
        v24 = 3;
        v22[0] = (__int64)"expected summary type";
        sub_11FD800(v2, v7, (__int64)v22, 1);
        result = 1;
        goto LABEL_9;
    }
    v11 = v8;
    sub_2240A30(v22);
    if ( v11 )
      goto LABEL_8;
  }
  while ( *(_DWORD *)(a1 + 240) == 4 && (unsigned __int8)sub_1205540(a1) );
  if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' here") )
  {
LABEL_8:
    result = 1;
    goto LABEL_9;
  }
  result = sub_120AFE0(a1, 13, "expected ')' here");
LABEL_9:
  if ( v19 != (_BYTE *)v21 )
  {
    v14 = result;
    j_j___libc_free_0(v19, v21[0] + 1LL);
    return v14;
  }
  return result;
}
