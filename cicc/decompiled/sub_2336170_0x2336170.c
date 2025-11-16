// Function: sub_2336170
// Address: 0x2336170
//
__int64 __fastcall sub_2336170(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  _DWORD *v6; // rcx
  unsigned __int64 v7; // rdi
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned int v14; // eax
  __int64 v15; // rdx
  unsigned int v16; // [rsp+8h] [rbp-D8h]
  _DWORD *v17; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int64 v18; // [rsp+18h] [rbp-C8h]
  __int64 v19; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v20; // [rsp+34h] [rbp-ACh] BYREF
  int v21; // [rsp+3Ch] [rbp-A4h]
  _DWORD *v22; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int64 v23; // [rsp+48h] [rbp-98h]
  unsigned __int64 v24[4]; // [rsp+50h] [rbp-90h] BYREF
  const char *v25; // [rsp+70h] [rbp-70h] BYREF
  __int64 v26; // [rsp+78h] [rbp-68h]
  _QWORD **v27; // [rsp+80h] [rbp-60h]
  __int64 v28; // [rsp+88h] [rbp-58h]
  char v29; // [rsp+90h] [rbp-50h]
  _QWORD v30[2]; // [rsp+98h] [rbp-48h] BYREF
  _QWORD *v31; // [rsp+A8h] [rbp-38h] BYREF

  v17 = (_DWORD *)a2;
  v18 = a3;
  sub_24671A0(&v20, 0, 0, 0, 0);
  if ( !v18 )
  {
LABEL_19:
    *(_BYTE *)(a1 + 16) = *(_BYTE *)(a1 + 16) & 0xFC | 2;
    *(_QWORD *)a1 = v20;
    *(_DWORD *)(a1 + 8) = v21;
    return a1;
  }
  while ( 1 )
  {
    v22 = 0;
    v23 = 0;
    LOBYTE(v25) = 59;
    v4 = sub_C931B0((__int64 *)&v17, &v25, 1u, 0);
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
      v8 = (__int64)v17 + v5;
      if ( v4 > v18 )
        v4 = v18;
    }
    v22 = v6;
    v23 = v4;
    v17 = (_DWORD *)v8;
    v18 = v7;
    if ( v4 == 7 )
    {
      if ( *v6 == 1868785010 && *((_WORD *)v6 + 2) == 25974 && *((_BYTE *)v6 + 6) == 114 )
      {
        LOBYTE(v21) = 1;
        goto LABEL_18;
      }
    }
    else if ( v4 == 6 && *v6 == 1852990827 && *((_WORD *)v6 + 2) == 27749 )
    {
      LOBYTE(v20) = 1;
      goto LABEL_18;
    }
    if ( (unsigned __int8)sub_95CB50((const void **)&v22, "track-origins=", 0xEu) )
      break;
    if ( v23 != 12 || *(_QWORD *)v22 != 0x68632D7265676165LL || v22[2] != 1936417637 )
    {
      v9 = sub_C63BB0();
      v26 = 45;
      v11 = v10;
      v16 = v9;
      v25 = "invalid MemorySanitizer pass parameter '{0}' ";
      goto LABEL_12;
    }
    BYTE1(v21) = 1;
    v7 = v18;
LABEL_18:
    if ( !v7 )
      goto LABEL_19;
  }
  if ( !sub_C93CC0((__int64)v22, v23, 0, (__int64 *)&v25) && v25 == (const char *)(int)v25 )
  {
    HIDWORD(v20) = (_DWORD)v25;
    v7 = v18;
    goto LABEL_18;
  }
  v14 = sub_C63BB0();
  v26 = 72;
  v11 = v15;
  v16 = v14;
  v25 = "invalid argument to MemorySanitizer pass track-origins parameter: '{0}' ";
LABEL_12:
  v29 = 1;
  v27 = &v31;
  v28 = 1;
  v30[1] = &v22;
  v30[0] = &unk_49DB108;
  v31 = v30;
  sub_23328D0((__int64)v24, (__int64)&v25);
  sub_23058C0(&v19, (__int64)v24, v16, v11);
  v12 = v19;
  *(_BYTE *)(a1 + 16) |= 3u;
  *(_QWORD *)a1 = v12 & 0xFFFFFFFFFFFFFFFELL;
  sub_2240A30(v24);
  return a1;
}
