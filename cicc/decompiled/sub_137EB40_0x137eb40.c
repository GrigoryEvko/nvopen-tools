// Function: sub_137EB40
// Address: 0x137eb40
//
__int64 __fastcall sub_137EB40(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  char v6; // dl
  __int64 v8; // rbx
  __int64 v9; // rax
  _QWORD *v10; // rax
  _QWORD v11[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v12[2]; // [rsp+10h] [rbp-70h] BYREF
  void *v13; // [rsp+20h] [rbp-60h] BYREF
  __int64 v14; // [rsp+28h] [rbp-58h]
  __int64 v15; // [rsp+30h] [rbp-50h]
  __int64 v16; // [rsp+38h] [rbp-48h]
  int v17; // [rsp+40h] [rbp-40h]
  _QWORD *v18; // [rsp+48h] [rbp-38h]

  v4 = a1 + 16;
  v5 = sub_157EBA0(a2);
  v6 = *(_BYTE *)(v5 + 16);
  if ( v6 == 26 )
  {
    if ( (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) == 3 )
    {
      *(_QWORD *)a1 = v4;
      *(_BYTE *)(a1 + 17) = 0;
      *(_QWORD *)(a1 + 8) = 1;
      *(_BYTE *)(a1 + 16) = a3 == 0 ? 84 : 70;
      return a1;
    }
    goto LABEL_3;
  }
  if ( v6 != 27 )
  {
LABEL_3:
    *(_QWORD *)a1 = v4;
    sub_137E9E0((__int64 *)a1, byte_3F871B3, (__int64)byte_3F871B3);
    return a1;
  }
  if ( a3 )
  {
    LOBYTE(v12[0]) = 0;
    v8 = (unsigned int)(2 * a3);
    v11[0] = v12;
    v11[1] = 0;
    v17 = 1;
    v16 = 0;
    v15 = 0;
    v14 = 0;
    v13 = &unk_49EFBE0;
    v18 = v11;
    if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
      v9 = *(_QWORD *)(v5 - 8);
    else
      v9 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
    sub_16A95F0(*(_QWORD *)(v9 + 24 * v8) + 24LL, &v13, 1);
    if ( v16 != v14 )
      sub_16E7BA0(&v13);
    v10 = v18;
    *(_QWORD *)a1 = v4;
    sub_137EA90((__int64 *)a1, (_BYTE *)*v10, *v10 + v10[1]);
    sub_16E7BC0(&v13);
    if ( (_QWORD *)v11[0] != v12 )
      j_j___libc_free_0(v11[0], v12[0] + 1LL);
  }
  else
  {
    *(_QWORD *)a1 = v4;
    *(_WORD *)(a1 + 16) = 25956;
    *(_BYTE *)(a1 + 18) = 102;
    *(_QWORD *)(a1 + 8) = 3;
    *(_BYTE *)(a1 + 19) = 0;
  }
  return a1;
}
