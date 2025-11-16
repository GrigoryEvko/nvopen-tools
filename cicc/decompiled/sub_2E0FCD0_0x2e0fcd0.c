// Function: sub_2E0FCD0
// Address: 0x2e0fcd0
//
void __fastcall sub_2E0FCD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r15
  __int128 v11; // rax
  _QWORD v12[4]; // [rsp+30h] [rbp-1E0h] BYREF
  _BYTE *v13; // [rsp+50h] [rbp-1C0h]
  __int64 v14; // [rsp+58h] [rbp-1B8h]
  _BYTE v15[432]; // [rsp+60h] [rbp-1B0h] BYREF

  v14 = 0x1000000000LL;
  v6 = *(unsigned int *)(a2 + 8);
  v7 = *(_QWORD *)a2;
  v12[0] = a1;
  v13 = v15;
  v12[1] = 0;
  v8 = v7 + 24 * v6;
  while ( v8 != v7 )
  {
    while ( *(_QWORD *)(v7 + 16) != a3 )
    {
      v7 += 24;
      if ( v8 == v7 )
        goto LABEL_6;
    }
    v11 = *(_OWORD *)v7;
    v7 += 24;
    sub_2E0F380((__int64)v12, a2, *((__int64 *)&v11 + 1), a4, a5, a6, v11, a4);
  }
LABEL_6:
  sub_2E0B930((__int64)v12, a2, a3, a4, a5);
  if ( v13 != v15 )
    _libc_free((unsigned __int64)v13);
}
