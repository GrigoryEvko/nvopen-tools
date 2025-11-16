// Function: sub_1299060
// Address: 0x1299060
//
__int64 __fastcall sub_1299060(_QWORD **a1, __int64 a2, unsigned __int8 a3, _DWORD *a4)
{
  __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // rax
  _BYTE *v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // r12
  __int64 v16; // rax
  __int64 v18; // [rsp+18h] [rbp-58h] BYREF
  __int64 v19; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v20; // [rsp+28h] [rbp-48h]
  _BYTE *v21; // [rsp+30h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 16);
  v20 = 0;
  v21 = 0;
  v6 = *(_DWORD *)(v5 + 12);
  v19 = 0;
  if ( v6 == 2 )
    sub_127B550("indirect return not supported!", a4, 1);
  if ( v6 > 2 )
  {
    if ( v6 != 3 )
      sub_127B550("unknown ABI variant for return type!", a4, 1);
    v7 = sub_1643270(**a1);
  }
  else
  {
    v7 = sub_127A030((__int64)a1, *(_QWORD *)(v5 + 24), 0);
  }
  v8 = *(_QWORD *)(a2 + 16);
  v9 = v8 + 40;
  v10 = v8 + 8 * (5LL * *(unsigned int *)(a2 + 8) + 5);
  if ( v10 != v8 + 40 )
  {
    do
    {
      v13 = *(_DWORD *)(v9 + 12);
      if ( v13 == 2 )
      {
        v16 = sub_127A040((__int64)a1, *(_QWORD *)(v9 + 24));
        v11 = sub_1646BA0(v16, 0);
        v12 = v20;
        v18 = v11;
        if ( v20 != v21 )
        {
LABEL_7:
          if ( v12 )
          {
            *(_QWORD *)v12 = v11;
            v12 = v20;
          }
          v20 = v12 + 8;
          goto LABEL_10;
        }
      }
      else
      {
        if ( v13 > 2 )
        {
          if ( v13 != 3 )
            sub_127B550("unknown ABI variant for argument!", a4, 1);
          goto LABEL_10;
        }
        v11 = sub_127A030((__int64)a1, *(_QWORD *)(v9 + 24), 0);
        v12 = v20;
        v18 = v11;
        if ( v20 != v21 )
          goto LABEL_7;
      }
      sub_1278040((__int64)&v19, v12, &v18);
LABEL_10:
      v9 += 40;
    }
    while ( v9 != v10 );
  }
  v14 = sub_1644EA0(v7, v19, (__int64)&v20[-v19] >> 3, a3);
  if ( v19 )
    j_j___libc_free_0(v19, &v21[-v19]);
  return v14;
}
