// Function: sub_2336CB0
// Address: 0x2336cb0
//
char *__fastcall sub_2336CB0(char *a1, __int64 a2, unsigned __int64 a3)
{
  char v3; // r14
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned __int64 v7; // r8
  __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  char v15; // al
  char v16; // [rsp+Fh] [rbp-C1h]
  __int64 v17; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int64 v18; // [rsp+18h] [rbp-B8h]
  __int64 v19; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v20; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v21; // [rsp+38h] [rbp-98h]
  unsigned __int64 v22[4]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v23[4]; // [rsp+60h] [rbp-70h] BYREF
  char v24; // [rsp+80h] [rbp-50h]
  _QWORD v25[2]; // [rsp+88h] [rbp-48h] BYREF
  _QWORD *v26; // [rsp+98h] [rbp-38h] BYREF

  v3 = 0;
  v17 = a2;
  v18 = a3;
  v16 = 0;
  if ( !a3 )
  {
LABEL_16:
    v15 = a1[8];
    a1[1] = v3;
    a1[8] = v15 & 0xFC | 2;
    *a1 = v16;
    return a1;
  }
  while ( 1 )
  {
    v20 = 0;
    v21 = 0;
    LOBYTE(v23[0]) = 59;
    v4 = sub_C931B0(&v17, v23, 1u, 0);
    if ( v4 == -1 )
    {
      v6 = v17;
      v4 = v18;
      v7 = 0;
      v8 = 0;
    }
    else
    {
      v5 = v4 + 1;
      v6 = v17;
      if ( v4 + 1 > v18 )
      {
        v5 = v18;
        v7 = 0;
      }
      else
      {
        v7 = v18 - v5;
      }
      v8 = v17 + v5;
      if ( v4 > v18 )
        v4 = v18;
    }
    v20 = v6;
    v21 = v4;
    v17 = v8;
    v18 = v7;
    if ( v4 != 7 )
      break;
    if ( *(_DWORD *)v6 != 1852401780 || *(_WORD *)(v6 + 4) != 29804 || *(_BYTE *)(v6 + 6) != 111 )
      goto LABEL_9;
    v16 = 1;
LABEL_15:
    if ( !v7 )
      goto LABEL_16;
  }
  if ( v4 == 12 && *(_QWORD *)v6 == 0x6D75732D74696D65LL && *(_DWORD *)(v6 + 8) == 2037539181 )
  {
    v3 = 1;
    goto LABEL_15;
  }
LABEL_9:
  v9 = sub_C63BB0();
  v23[1] = 42;
  v10 = v9;
  v12 = v11;
  v23[0] = "invalid EmbedBitcode pass parameter '{0}' ";
  v23[2] = &v26;
  v23[3] = 1;
  v24 = 1;
  v25[0] = &unk_49DB108;
  v25[1] = &v20;
  v26 = v25;
  sub_23328D0((__int64)v22, (__int64)v23);
  sub_23058C0(&v19, (__int64)v22, v10, v12);
  v13 = v19;
  a1[8] |= 3u;
  *(_QWORD *)a1 = v13 & 0xFFFFFFFFFFFFFFFELL;
  sub_2240A30(v22);
  return a1;
}
