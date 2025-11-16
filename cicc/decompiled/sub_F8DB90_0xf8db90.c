// Function: sub_F8DB90
// Address: 0xf8db90
//
_QWORD *__fastcall sub_F8DB90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int16 a5)
{
  __int64 v7; // rax
  __int64 v8; // r8
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  int v11; // ecx
  _QWORD *v12; // rdx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // [rsp-60h] [rbp-60h]
  __int64 v17[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( !a4 )
    BUG();
  v7 = *(_QWORD *)(a4 + 16);
  *(_QWORD *)(a1 + 568) = v7;
  *(_QWORD *)(a1 + 576) = a4;
  *(_WORD *)(a1 + 584) = a5;
  if ( a4 != v7 + 48 )
  {
    v17[0] = *(_QWORD *)sub_B46C60(a4 - 24);
    if ( v17[0] && (sub_B96E90((__int64)v17, v17[0], 1), (v8 = v17[0]) != 0) )
    {
      v9 = *(unsigned int *)(a1 + 528);
      v10 = *(_QWORD **)(a1 + 520);
      v11 = *(_DWORD *)(a1 + 528);
      v12 = &v10[2 * v9];
      if ( v10 != v12 )
      {
        while ( *(_DWORD *)v10 )
        {
          v10 += 2;
          if ( v12 == v10 )
            goto LABEL_15;
        }
        v10[1] = v17[0];
LABEL_10:
        sub_B91220((__int64)v17, v8);
        return sub_F8DB50((__int64 *)a1, a2, a3);
      }
LABEL_15:
      v14 = *(unsigned int *)(a1 + 532);
      if ( v9 >= v14 )
      {
        v15 = v9 + 1;
        if ( v14 < v15 )
        {
          v16 = v17[0];
          sub_C8D5F0(a1 + 520, (const void *)(a1 + 536), v15, 0x10u, v17[0], a1 + 536);
          v8 = v16;
          v12 = (_QWORD *)(*(_QWORD *)(a1 + 520) + 16LL * *(unsigned int *)(a1 + 528));
        }
        *v12 = 0;
        v12[1] = v8;
        v8 = v17[0];
        ++*(_DWORD *)(a1 + 528);
      }
      else
      {
        if ( v12 )
        {
          *(_DWORD *)v12 = 0;
          v12[1] = v8;
          v8 = v17[0];
          v11 = *(_DWORD *)(a1 + 528);
        }
        *(_DWORD *)(a1 + 528) = v11 + 1;
      }
    }
    else
    {
      sub_93FB40(a1 + 520, 0);
      v8 = v17[0];
    }
    if ( !v8 )
      return sub_F8DB50((__int64 *)a1, a2, a3);
    goto LABEL_10;
  }
  return sub_F8DB50((__int64 *)a1, a2, a3);
}
