// Function: sub_B6E5C0
// Address: 0xb6e5c0
//
__int64 __fastcall sub_B6E5C0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  _QWORD v4[2]; // [rsp+0h] [rbp-A0h] BYREF
  _QWORD v5[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v6[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v7[3]; // [rsp+30h] [rbp-70h] BYREF
  _BYTE *v8; // [rsp+48h] [rbp-58h]
  _BYTE *v9; // [rsp+50h] [rbp-50h]
  __int64 v10; // [rsp+58h] [rbp-48h]
  _QWORD *v11; // [rsp+60h] [rbp-40h]

  if ( (unsigned int)(*(_DWORD *)(a2 + 8) - 13) > 8
    || (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a2 + 32LL))(a2) )
  {
    LOBYTE(v6[0]) = 0;
    v5[1] = 0;
    v10 = 0x100000000LL;
    v7[0] = &unk_49DD210;
    v5[0] = v6;
    v7[1] = 0;
    v7[2] = 0;
    v8 = 0;
    v9 = 0;
    v11 = v5;
    sub_CB5980(v7, 0, 0, 0);
    v4[1] = v7;
    v4[0] = &unk_49E1428;
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a2 + 24LL))(a2, v4);
    if ( v8 == v9 )
      sub_CB6200(v7, "\n", 1);
    else
      *v9++ = 10;
    v2 = 256;
    LOBYTE(v2) = *(_BYTE *)(a2 + 12);
    sub_CEB020(v5, v2, 1);
    v7[0] = &unk_49DD210;
    sub_CB5840(v7);
    if ( (_QWORD *)v5[0] != v6 )
      j_j___libc_free_0(v5[0], v6[0] + 1LL);
  }
  return 1;
}
