// Function: sub_2335980
// Address: 0x2335980
//
__int64 __fastcall sub_2335980(__int64 a1, const void *a2, size_t a3)
{
  int v4; // edx
  unsigned int v6; // ebx
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 *v10; // rdi
  _QWORD v11[3]; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v12; // [rsp+18h] [rbp-98h] BYREF
  __int64 v13; // [rsp+24h] [rbp-8Ch]
  int v14; // [rsp+2Ch] [rbp-84h]
  _QWORD v15[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v16; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v17[4]; // [rsp+50h] [rbp-60h] BYREF
  char v18; // [rsp+70h] [rbp-40h]
  _QWORD v19[2]; // [rsp+78h] [rbp-38h] BYREF
  _QWORD *v20; // [rsp+88h] [rbp-28h] BYREF

  v11[0] = a2;
  v11[1] = a3;
  v13 = sub_232E370(a2, a3);
  v14 = v4;
  if ( !(_BYTE)v4 || qword_5033F08 == v13 )
  {
    v17[1] = 48;
    v6 = sub_C63BB0();
    v8 = v7;
    v17[0] = "invalid function-simplification parameter '{0}' ";
    v17[2] = &v20;
    v17[3] = 1;
    v18 = 1;
    v19[0] = &unk_49DB108;
    v19[1] = v11;
    v20 = v19;
    sub_23328D0((__int64)v15, (__int64)v17);
    sub_23058C0(&v12, (__int64)v15, v6, v8);
    v9 = v12;
    v10 = (__int64 *)v15[0];
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v9 & 0xFFFFFFFFFFFFFFFELL;
    if ( v10 != &v16 )
      j_j___libc_free_0((unsigned __int64)v10);
  }
  else
  {
    *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
    *(_QWORD *)a1 = v13;
  }
  return a1;
}
