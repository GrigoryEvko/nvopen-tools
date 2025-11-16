// Function: sub_2336EE0
// Address: 0x2336ee0
//
__int64 __fastcall sub_2336EE0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  _WORD *v6; // rcx
  unsigned __int64 v7; // rdi
  __int64 v8; // rdx
  char v9; // dl
  char v10; // al
  char v11; // al
  unsigned int v13; // eax
  unsigned int v14; // ebx
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  int v18; // [rsp+8h] [rbp-C8h]
  char v19; // [rsp+Ch] [rbp-C4h]
  char v20; // [rsp+Dh] [rbp-C3h]
  char v21; // [rsp+Eh] [rbp-C2h]
  char v22; // [rsp+Fh] [rbp-C1h]
  _WORD *v23; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int64 v24; // [rsp+18h] [rbp-B8h]
  __int64 v25; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v26; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v27; // [rsp+38h] [rbp-98h]
  unsigned __int64 v28[4]; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v29[4]; // [rsp+60h] [rbp-70h] BYREF
  char v30; // [rsp+80h] [rbp-50h]
  _QWORD v31[2]; // [rsp+88h] [rbp-48h] BYREF
  _QWORD *v32; // [rsp+98h] [rbp-38h] BYREF

  v23 = (_WORD *)a2;
  v24 = a3;
  v19 = 1;
  v20 = 0;
  v21 = 1;
  v22 = 1;
  v18 = 0;
  if ( a3 )
  {
    while ( 1 )
    {
      v26 = 0;
      v27 = 0;
      LOBYTE(v29[0]) = 59;
      v4 = sub_C931B0((__int64 *)&v23, v29, 1u, 0);
      if ( v4 == -1 )
      {
        v6 = v23;
        v4 = v24;
        v7 = 0;
        v8 = 0;
      }
      else
      {
        v5 = v4 + 1;
        v6 = v23;
        if ( v4 + 1 > v24 )
        {
          v5 = v24;
          v7 = 0;
        }
        else
        {
          v7 = v24 - v5;
        }
        v8 = (__int64)v23 + v5;
        if ( v4 > v24 )
          v4 = v24;
      }
      v23 = (_WORD *)v8;
      v9 = 1;
      v26 = (__int64)v6;
      v27 = v4;
      v24 = v7;
      if ( v4 > 2 )
      {
        if ( *v6 == 28526 && *((_BYTE *)v6 + 2) == 45 )
        {
          v4 -= 3LL;
          v6 = (_WORD *)((char *)v6 + 3);
          v9 = 0;
          v26 = (__int64)v6;
          v27 = v4;
        }
        else
        {
          v9 = 1;
        }
        if ( v4 == 12 )
        {
          if ( *(_QWORD *)v6 != 0x79622D70756F7267LL || *((_DWORD *)v6 + 2) != 1702065453 )
            goto LABEL_10;
          v22 = v9;
          if ( !v7 )
            break;
        }
        else
        {
          if ( v4 != 17 )
            goto LABEL_8;
          if ( *(_QWORD *)v6 ^ 0x732D65726F6E6769LL | *((_QWORD *)v6 + 1) ^ 0x73752D656C676E69LL
            || *((_BYTE *)v6 + 16) != 101 )
          {
            goto LABEL_10;
          }
          v21 = v9;
          if ( !v7 )
            break;
        }
      }
      else
      {
LABEL_8:
        if ( v4 == 11 )
        {
          if ( *(_QWORD *)v6 == 0x6F632D656772656DLL && v6[4] == 29550 && *((_BYTE *)v6 + 10) == 116 )
          {
            v20 = v9;
            goto LABEL_11;
          }
        }
        else if ( v4 == 14
               && *(_QWORD *)v6 == 0x78652D656772656DLL
               && *((_DWORD *)v6 + 2) == 1852990836
               && v6[6] == 27745 )
        {
          v19 = v9;
          goto LABEL_11;
        }
LABEL_10:
        v10 = sub_95CB50((const void **)&v26, "max-offset=", 0xBu);
        v7 = v24;
        if ( v10 )
        {
          if ( sub_C93C90(v26, v27, 0, v29) || v29[0] != LODWORD(v29[0]) )
          {
            v13 = sub_C63BB0();
            v29[1] = 40;
            v14 = v13;
            v16 = v15;
            v29[0] = (unsigned __int64)"invalid GlobalMergePass parameter '{0}' ";
            v29[2] = (unsigned __int64)&v32;
            v29[3] = 1;
            v30 = 1;
            v31[0] = &unk_49DB108;
            v31[1] = &v26;
            v32 = v31;
            sub_23328D0((__int64)v28, (__int64)v29);
            sub_23058C0(&v25, (__int64)v28, v14, v16);
            v17 = v25;
            *(_BYTE *)(a1 + 16) |= 3u;
            *(_QWORD *)a1 = v17 & 0xFFFFFFFFFFFFFFFELL;
            sub_2240A30(v28);
            return a1;
          }
          v18 = v29[0];
          v7 = v24;
        }
LABEL_11:
        if ( !v7 )
          break;
      }
    }
  }
  v11 = *(_BYTE *)(a1 + 16);
  *(_DWORD *)(a1 + 4) = 0;
  *(_BYTE *)(a1 + 14) = 0;
  *(_BYTE *)(a1 + 16) = v11 & 0xFC | 2;
  *(_DWORD *)a1 = v18;
  *(_BYTE *)(a1 + 8) = v22;
  *(_BYTE *)(a1 + 9) = v21;
  *(_BYTE *)(a1 + 10) = v20;
  *(_BYTE *)(a1 + 11) = v19;
  *(_WORD *)(a1 + 12) = 0;
  return a1;
}
