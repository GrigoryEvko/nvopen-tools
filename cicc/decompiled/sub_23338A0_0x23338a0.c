// Function: sub_23338A0
// Address: 0x23338a0
//
_BYTE *__fastcall sub_23338A0(_BYTE *a1, _WORD *a2, size_t a3)
{
  char v3; // al
  char v5; // al
  unsigned int v6; // ebx
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rax
  const void *v10; // [rsp+0h] [rbp-A0h] BYREF
  size_t v11; // [rsp+8h] [rbp-98h]
  __int64 v12; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v13[4]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v14[4]; // [rsp+40h] [rbp-60h] BYREF
  char v15; // [rsp+60h] [rbp-40h]
  _QWORD v16[2]; // [rsp+68h] [rbp-38h] BYREF
  _QWORD *v17; // [rsp+78h] [rbp-28h] BYREF

  v10 = a2;
  v11 = a3;
  if ( !a3 || a3 == 10 && *(_QWORD *)a2 == 0x632D796669646F6DLL && a2[4] == 26470 )
  {
    v5 = a1[8];
    *a1 = 0;
    a1[8] = v5 & 0xFC | 2;
  }
  else
  {
    if ( sub_9691B0(v10, v11, "preserve-cfg", 12) )
    {
      v3 = a1[8];
      *a1 = 1;
      a1[8] = v3 & 0xFC | 2;
      return a1;
    }
    v14[1] = 86;
    v6 = sub_C63BB0();
    v8 = v7;
    v14[0] = "invalid SROA pass parameter '{0}' (either preserve-cfg or modify-cfg can be specified)";
    v14[2] = &v17;
    v14[3] = 1;
    v15 = 1;
    v16[0] = &unk_49DB108;
    v16[1] = &v10;
    v17 = v16;
    sub_23328D0((__int64)v13, (__int64)v14);
    sub_23058C0(&v12, (__int64)v13, v6, v8);
    v9 = v12;
    a1[8] |= 3u;
    *(_QWORD *)a1 = v9 & 0xFFFFFFFFFFFFFFFELL;
    sub_2240A30(v13);
  }
  return a1;
}
