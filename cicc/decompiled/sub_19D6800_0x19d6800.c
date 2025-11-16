// Function: sub_19D6800
// Address: 0x19d6800
//
__int64 __fastcall sub_19D6800(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __int64 v3; // rax
  int v4; // edx
  __int64 v5; // rbx
  unsigned __int64 v6; // r13
  bool v8; // cc
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-48h]
  unsigned __int64 v13; // [rsp+10h] [rbp-40h]
  __int64 v14; // [rsp+18h] [rbp-38h]
  __int64 v15; // [rsp+20h] [rbp-30h] BYREF
  __int64 v16; // [rsp+28h] [rbp-28h]

  v2 = *(_BYTE *)(a2 + 16) == 54;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 1;
  if ( !v2 )
  {
    v3 = 0;
    v4 = 1;
    v5 = 0;
    v6 = 0;
LABEL_3:
    *(_QWORD *)a1 = v6;
    *(_QWORD *)(a1 + 8) = v5;
    *(_DWORD *)(a1 + 24) = v4;
    *(_QWORD *)(a1 + 16) = v3;
    return a1;
  }
  v5 = a2;
  if ( !(unsigned __int8)sub_15F2E00(a2, *(_QWORD *)(a2 + 40)) && (*(_BYTE *)(a2 + 18) & 1) == 0 )
  {
    v6 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v6 + 16) == 56 )
    {
      if ( (unsigned __int8)sub_15F2E00(a2, *(_QWORD *)(a2 + 40)) )
        goto LABEL_7;
      v9 = sub_15F2050(v6);
      v10 = sub_1632FA0(v9);
      if ( !(unsigned __int8)sub_13F8680(v6, v10, 0, 0) )
        goto LABEL_7;
      v12 = sub_15A9570(v10, *(_QWORD *)v6);
      if ( v12 > 0x40 )
        sub_16A4EF0((__int64)&v11, 0, 0);
      else
        v11 = 0;
      if ( (unsigned int)v16 > 0x40 && v15 )
        j_j___libc_free_0_0(v15);
      v15 = v11;
      LODWORD(v16) = v12;
      if ( (unsigned __int8)sub_15FA310(v6, v10, (__int64)&v15) )
      {
        v4 = v16;
        v3 = v15;
        goto LABEL_3;
      }
    }
    v6 = v13;
    v5 = v14;
    v4 = v16;
    v3 = v15;
    goto LABEL_3;
  }
LABEL_7:
  v8 = (unsigned int)v16 <= 0x40;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  if ( v8 || !v15 )
    return a1;
  j_j___libc_free_0_0(v15);
  return a1;
}
