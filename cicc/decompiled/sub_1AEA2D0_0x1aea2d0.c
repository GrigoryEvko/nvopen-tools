// Function: sub_1AEA2D0
// Address: 0x1aea2d0
//
void __fastcall sub_1AEA2D0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // edx
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r15
  _BYTE *v8; // rdi
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // [rsp+0h] [rbp-60h]
  _BYTE *v16; // [rsp+10h] [rbp-50h] BYREF
  __int64 v17; // [rsp+18h] [rbp-48h]
  _BYTE v18[64]; // [rsp+20h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a1 + 20);
  v16 = v18;
  v5 = v4 & 0xFFFFFFF;
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 24 * (1 - v5)) + 24LL);
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 24 * (2 - v5)) + 24LL);
  v17 = 0x100000000LL;
  sub_1AEA1F0((__int64)&v16, (__int64)a2);
  v8 = &v16[8 * (unsigned int)v17];
  if ( v16 == v8 )
  {
LABEL_9:
    if ( v16 != v18 )
      _libc_free((unsigned __int64)v16);
    if ( (unsigned __int8)sub_1AE93B0(*a2, a1) )
    {
      v14 = a2[5];
      v11 = sub_157EE30(v14);
      v12 = v11;
      if ( v11 )
      {
        if ( v11 == v14 + 40 )
          return;
        v12 = v11 - 24;
      }
      v13 = sub_15C70A0(a1 + 48);
      sub_15A76D0(a3, (__int64)a2, v6, v7, v13, v12);
    }
  }
  else
  {
    v9 = (unsigned __int64)v16;
    while ( 1 )
    {
      v10 = *(_DWORD *)(*(_QWORD *)v9 + 20LL) & 0xFFFFFFF;
      if ( v6 == *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v9 + 24 * (1 - v10)) + 24LL)
        && v7 == *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v9 + 24 * (2 - v10)) + 24LL) )
      {
        break;
      }
      v9 += 8LL;
      if ( v8 == (_BYTE *)v9 )
        goto LABEL_9;
    }
    if ( v16 != v18 )
      _libc_free((unsigned __int64)v16);
  }
}
