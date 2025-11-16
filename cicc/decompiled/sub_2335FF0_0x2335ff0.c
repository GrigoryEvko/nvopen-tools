// Function: sub_2335FF0
// Address: 0x2335ff0
//
__int64 __fastcall sub_2335FF0(__int64 a1, _QWORD *a2, size_t a3)
{
  char v3; // al
  char v5; // al
  char v6; // al
  unsigned int v7; // ebx
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // rax
  const void *v11; // [rsp+0h] [rbp-A0h] BYREF
  size_t v12; // [rsp+8h] [rbp-98h]
  __int64 v13; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v14[4]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v15[4]; // [rsp+40h] [rbp-60h] BYREF
  char v16; // [rsp+60h] [rbp-40h]
  _QWORD v17[2]; // [rsp+68h] [rbp-38h] BYREF
  _QWORD *v18; // [rsp+78h] [rbp-28h] BYREF

  v11 = a2;
  v12 = a3;
  if ( !a3 )
  {
    v3 = *(_BYTE *)(a1 + 8);
    *(_DWORD *)a1 = 0;
    *(_BYTE *)(a1 + 8) = v3 & 0xFC | 2;
    return a1;
  }
  if ( a3 == 8 && *a2 == 0x64656C6961746564LL )
  {
    v6 = *(_BYTE *)(a1 + 8);
    *(_DWORD *)a1 = 1;
    *(_BYTE *)(a1 + 8) = v6 & 0xFC | 2;
    return a1;
  }
  else
  {
    if ( !sub_9691B0(v11, v12, "call-target-ignored", 19) )
    {
      v15[1] = 48;
      v7 = sub_C63BB0();
      v9 = v8;
      v15[0] = "invalid structural hash printer parameter '{0}' ";
      v15[2] = &v18;
      v15[3] = 1;
      v16 = 1;
      v17[0] = &unk_49DB108;
      v17[1] = &v11;
      v18 = v17;
      sub_23328D0((__int64)v14, (__int64)v15);
      sub_23058C0(&v13, (__int64)v14, v7, v9);
      v10 = v13;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v10 & 0xFFFFFFFFFFFFFFFELL;
      sub_2240A30(v14);
      return a1;
    }
    v5 = *(_BYTE *)(a1 + 8);
    *(_DWORD *)a1 = 2;
    *(_BYTE *)(a1 + 8) = v5 & 0xFC | 2;
    return a1;
  }
}
