// Function: sub_2332EB0
// Address: 0x2332eb0
//
__int64 __fastcall sub_2332EB0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned __int64 v7; // rdi
  __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  char v15; // si
  int v16; // [rsp+4h] [rbp-CCh]
  int v17; // [rsp+8h] [rbp-C8h]
  char v18; // [rsp+Eh] [rbp-C2h]
  char v19; // [rsp+Fh] [rbp-C1h]
  __int64 v20; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int64 v21; // [rsp+18h] [rbp-B8h]
  __int64 v22; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v23; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v24; // [rsp+38h] [rbp-98h]
  unsigned __int64 v25[4]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v26[4]; // [rsp+60h] [rbp-70h] BYREF
  char v27; // [rsp+80h] [rbp-50h]
  _QWORD v28[2]; // [rsp+88h] [rbp-48h] BYREF
  _QWORD *v29; // [rsp+98h] [rbp-38h] BYREF

  v20 = a2;
  v21 = a3;
  v18 = 0;
  v19 = 1;
  v16 = qword_4FFDE88[8];
  v17 = qword_4FFDDA8[8];
  if ( a3 )
  {
    while ( 1 )
    {
      v23 = 0;
      v24 = 0;
      LOBYTE(v26[0]) = 59;
      v4 = sub_C931B0(&v20, v26, 1u, 0);
      if ( v4 == -1 )
      {
        v6 = v20;
        v4 = v21;
        v7 = 0;
        v8 = 0;
      }
      else
      {
        v5 = v4 + 1;
        v6 = v20;
        if ( v4 + 1 > v21 )
        {
          v5 = v21;
          v7 = 0;
        }
        else
        {
          v7 = v21 - v5;
        }
        v8 = v20 + v5;
        if ( v4 > v21 )
          v4 = v21;
      }
      v23 = v6;
      v24 = v4;
      v20 = v8;
      v21 = v7;
      if ( v4 <= 2 )
        break;
      if ( *(_WORD *)v6 == 28526 && *(_BYTE *)(v6 + 2) == 45 )
      {
        v4 -= 3LL;
        v6 += 3;
        v15 = 0;
        v23 = v6;
        v24 = v4;
      }
      else
      {
        v15 = 1;
      }
      if ( v4 == 16 )
      {
        if ( *(_QWORD *)v6 ^ 0x657073776F6C6C61LL | *(_QWORD *)(v6 + 8) ^ 0x6E6F6974616C7563LL )
          break;
        v19 = v15;
      }
      else
      {
        if ( v4 != 18
          || *(_QWORD *)v6 ^ 0x61767265736E6F63LL | *(_QWORD *)(v6 + 8) ^ 0x6C61632D65766974LL
          || *(_WORD *)(v6 + 16) != 29548 )
        {
          break;
        }
        v18 = v15;
      }
      if ( !v7 )
        goto LABEL_17;
    }
    v9 = sub_C63BB0();
    v26[1] = 34;
    v10 = v9;
    v12 = v11;
    v26[0] = "invalid LICM pass parameter '{0}' ";
    v26[2] = &v29;
    v26[3] = 1;
    v27 = 1;
    v28[0] = &unk_49DB108;
    v28[1] = &v23;
    v29 = v28;
    sub_23328D0((__int64)v25, (__int64)v26);
    sub_23058C0(&v22, (__int64)v25, v10, v12);
    v13 = v22;
    *(_BYTE *)(a1 + 16) |= 3u;
    *(_QWORD *)a1 = v13 & 0xFFFFFFFFFFFFFFFELL;
    sub_2240A30(v25);
  }
  else
  {
LABEL_17:
    *(_BYTE *)(a1 + 16) = *(_BYTE *)(a1 + 16) & 0xFC | 2;
    *(_DWORD *)a1 = v16;
    *(_DWORD *)(a1 + 4) = v17;
    *(_BYTE *)(a1 + 8) = v19;
    *(_BYTE *)(a1 + 9) = v18;
  }
  return a1;
}
