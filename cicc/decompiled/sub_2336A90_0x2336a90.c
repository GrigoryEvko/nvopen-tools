// Function: sub_2336A90
// Address: 0x2336a90
//
char *__fastcall sub_2336A90(char *a1, __int64 a2, unsigned __int64 a3)
{
  char v3; // r14
  char v4; // r13
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned int v11; // eax
  unsigned int v12; // ebx
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rax
  char v17; // al
  __int64 v18; // [rsp+0h] [rbp-C0h] BYREF
  unsigned __int64 v19; // [rsp+8h] [rbp-B8h]
  __int64 v20; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v21; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 v22; // [rsp+28h] [rbp-98h]
  unsigned __int64 v23[4]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v24[4]; // [rsp+50h] [rbp-70h] BYREF
  char v25; // [rsp+70h] [rbp-50h]
  _QWORD v26[2]; // [rsp+78h] [rbp-48h] BYREF
  _QWORD *v27; // [rsp+88h] [rbp-38h] BYREF

  v3 = 0;
  v4 = 0;
  v18 = a2;
  v19 = a3;
  if ( !a3 )
  {
LABEL_16:
    v17 = a1[8];
    *a1 = v4;
    a1[1] = v3;
    a1[2] = 0;
    a1[8] = v17 & 0xFC | 2;
    return a1;
  }
  while ( 1 )
  {
    v21 = 0;
    v22 = 0;
    LOBYTE(v24[0]) = 59;
    v6 = sub_C931B0(&v18, v24, 1u, 0);
    if ( v6 == -1 )
    {
      v8 = v18;
      v6 = v19;
      v9 = 0;
      v10 = 0;
    }
    else
    {
      v7 = v6 + 1;
      v8 = v18;
      if ( v6 + 1 > v19 )
      {
        v7 = v19;
        v9 = 0;
      }
      else
      {
        v9 = v19 - v7;
      }
      v10 = v18 + v7;
      if ( v6 > v19 )
        v6 = v19;
    }
    v21 = v8;
    v22 = v6;
    v18 = v10;
    v19 = v9;
    if ( v6 != 7 )
      break;
    if ( *(_DWORD *)v8 != 1868785010 || *(_WORD *)(v8 + 4) != 25974 || *(_BYTE *)(v8 + 6) != 114 )
      goto LABEL_9;
    v3 = 1;
LABEL_15:
    if ( !v9 )
      goto LABEL_16;
  }
  if ( v6 == 6 && *(_DWORD *)v8 == 1852990827 && *(_WORD *)(v8 + 4) == 27749 )
  {
    v4 = 1;
    goto LABEL_15;
  }
LABEL_9:
  v11 = sub_C63BB0();
  v24[1] = 48;
  v12 = v11;
  v14 = v13;
  v24[0] = "invalid HWAddressSanitizer pass parameter '{0}' ";
  v24[2] = &v27;
  v24[3] = 1;
  v25 = 1;
  v26[0] = &unk_49DB108;
  v26[1] = &v21;
  v27 = v26;
  sub_23328D0((__int64)v23, (__int64)v24);
  sub_23058C0(&v20, (__int64)v23, v12, v14);
  v15 = v20;
  a1[8] |= 3u;
  *(_QWORD *)a1 = v15 & 0xFFFFFFFFFFFFFFFELL;
  sub_2240A30(v23);
  return a1;
}
