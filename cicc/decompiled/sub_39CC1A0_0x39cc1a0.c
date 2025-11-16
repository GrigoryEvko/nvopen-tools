// Function: sub_39CC1A0
// Address: 0x39cc1a0
//
__int64 __fastcall sub_39CC1A0(__int64 *a1, __int64 a2)
{
  __int16 v4; // r14
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned __int8 *v7; // rsi
  unsigned __int8 v8; // al
  __int64 v9; // r14
  __int64 v10; // rdi
  void *v11; // rax
  size_t v12; // rdx

  v4 = *(_WORD *)(a2 + 2);
  v5 = sub_145CDC0(0x30u, a1 + 11);
  v6 = v5;
  if ( v5 )
  {
    *(_WORD *)(v5 + 28) = v4;
    *(_QWORD *)v5 = v5 | 4;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_DWORD *)(v5 + 24) = -1;
    *(_BYTE *)(v5 + 30) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0;
  }
  sub_39A55B0((__int64)a1, (unsigned __int8 *)a2, (unsigned __int8 *)v5);
  v7 = *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)));
  v8 = *v7;
  if ( *v7 != 20 )
  {
    if ( v8 == 21 )
    {
      v9 = sub_39A8430(a1, (__int64)v7);
      goto LABEL_5;
    }
    if ( v8 == 17 )
    {
      v9 = sub_39A8220((__int64)a1, (__int64)v7, 0);
      goto LABEL_5;
    }
    if ( v8 > 0xEu )
    {
      if ( (unsigned __int8)(v8 - 32) > 1u )
      {
        if ( v8 == 24 )
        {
          v9 = (__int64)sub_39CBE70(a1, (__int64)v7, 0, 0);
          goto LABEL_5;
        }
        goto LABEL_15;
      }
    }
    else if ( v8 <= 0xAu )
    {
LABEL_15:
      v9 = (__int64)sub_39A23D0((__int64)a1, v7);
      goto LABEL_5;
    }
    v9 = sub_39A64F0(a1, (__int64)v7);
    goto LABEL_5;
  }
  v9 = sub_39A82F0(a1, (__int64)v7);
LABEL_5:
  sub_39A36D0((__int64)a1, v6, *(_DWORD *)(a2 + 24), *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8))));
  sub_39A3B20((__int64)a1, v6, 24, v9);
  v10 = *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
  if ( v10 )
  {
    v11 = (void *)sub_161E970(v10);
    if ( v12 )
      sub_39A3F30(a1, v6, 3, v11, v12);
  }
  return v6;
}
