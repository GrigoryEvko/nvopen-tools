// Function: sub_2335CE0
// Address: 0x2335ce0
//
char *__fastcall sub_2335CE0(char *a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdx
  __int64 v5; // rcx
  unsigned __int64 v6; // r8
  __int64 v7; // rdx
  unsigned int v8; // eax
  unsigned int v9; // ebx
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rax
  char v14; // [rsp+Eh] [rbp-C2h]
  char v15; // [rsp+Fh] [rbp-C1h]
  __int64 v16; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int64 v17; // [rsp+18h] [rbp-B8h]
  __int64 v18; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v19; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v20; // [rsp+38h] [rbp-98h]
  unsigned __int64 v21[4]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v22[4]; // [rsp+60h] [rbp-70h] BYREF
  char v23; // [rsp+80h] [rbp-50h]
  _QWORD v24[2]; // [rsp+88h] [rbp-48h] BYREF
  _QWORD *v25; // [rsp+98h] [rbp-38h] BYREF

  v16 = a2;
  v17 = a3;
  v15 = 1;
  v14 = 1;
  if ( !a3 )
  {
LABEL_24:
    a1[8] = a1[8] & 0xFC | 2;
    *a1 = v15;
    a1[1] = v14;
    return a1;
  }
  while ( 1 )
  {
    while ( 1 )
    {
      v19 = 0;
      v20 = 0;
      LOBYTE(v22[0]) = 59;
      v3 = sub_C931B0(&v16, v22, 1u, 0);
      if ( v3 == -1 )
      {
        v5 = v16;
        v3 = v17;
        v6 = 0;
        v7 = 0;
      }
      else
      {
        v4 = v3 + 1;
        v5 = v16;
        if ( v3 + 1 > v17 )
        {
          v4 = v17;
          v6 = 0;
        }
        else
        {
          v6 = v17 - v4;
        }
        v7 = v16 + v4;
        if ( v3 > v17 )
          v3 = v17;
      }
      v19 = v5;
      v20 = v3;
      v16 = v7;
      v17 = v6;
      if ( v3 != 17 )
        break;
      if ( !(*(_QWORD *)v5 ^ 0x732D796669646F6DLL | *(_QWORD *)(v5 + 8) ^ 0x656D2D6465726168LL)
        && *(_BYTE *)(v5 + 16) == 109 )
      {
        v15 = 1;
        goto LABEL_30;
      }
      if ( *(_QWORD *)v5 ^ 0x672D796669646F6DLL | *(_QWORD *)(v5 + 8) ^ 0x656D2D6C61626F6CLL
        || *(_BYTE *)(v5 + 16) != 109 )
      {
        goto LABEL_11;
      }
      v14 = 1;
      if ( !v6 )
        goto LABEL_24;
    }
    if ( v3 != 15 )
      break;
    if ( *(_QWORD *)v5 == 0x6168732D70696B73LL
      && *(_DWORD *)(v5 + 8) == 761554290
      && *(_WORD *)(v5 + 12) == 25965
      && *(_BYTE *)(v5 + 14) == 109 )
    {
      v15 = 0;
      goto LABEL_30;
    }
    if ( *(_QWORD *)v5 != 0x6F6C672D70696B73LL
      || *(_DWORD *)(v5 + 8) != 762077538
      || *(_WORD *)(v5 + 12) != 25965
      || *(_BYTE *)(v5 + 14) != 109 )
    {
      break;
    }
    v14 = 0;
LABEL_30:
    if ( !v6 )
      goto LABEL_24;
  }
LABEL_11:
  v8 = sub_C63BB0();
  v22[1] = 53;
  v9 = v8;
  v11 = v10;
  v22[0] = "invalid SetGlobalArrayAlignment pass parameter '{0}' ";
  v22[2] = &v25;
  v22[3] = 1;
  v23 = 1;
  v24[0] = &unk_49DB108;
  v24[1] = &v19;
  v25 = v24;
  sub_23328D0((__int64)v21, (__int64)v22);
  sub_23058C0(&v18, (__int64)v21, v9, v11);
  v12 = v18;
  a1[8] |= 3u;
  *(_QWORD *)a1 = v12 & 0xFFFFFFFFFFFFFFFELL;
  sub_2240A30(v21);
  return a1;
}
