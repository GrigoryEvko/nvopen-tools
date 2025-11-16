// Function: sub_C22680
// Address: 0xc22680
//
__int64 __fastcall sub_C22680(__int64 a1, __int64 a2)
{
  int v3; // eax
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // r13
  __int64 *v6; // r14
  __int64 *v7; // rax
  __int64 *v8; // r15
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  int v11; // ecx
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned __int64 v15; // [rsp+8h] [rbp-F8h]
  __int64 v16; // [rsp+8h] [rbp-F8h]
  __int64 v17; // [rsp+18h] [rbp-E8h] BYREF
  unsigned __int64 v18; // [rsp+20h] [rbp-E0h] BYREF
  __int64 *v19; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int64 v20; // [rsp+38h] [rbp-C8h]
  char v21; // [rsp+40h] [rbp-C0h]

  if ( !*(_BYTE *)(a2 + 178) )
  {
    sub_C21FD0((__int64)&v19, (_QWORD *)a2, &v17);
    if ( (v21 & 1) == 0 || (v3 = (int)v19, v4 = v20, !(_DWORD)v19) )
    {
      v13 = *(_QWORD *)(a2 + 296);
      v8 = v19;
      v9 = v20;
      v14 = 8 * v17;
      v10 = *(_QWORD *)(v13 + 8 * v17);
      if ( v10 )
      {
        v11 = 0;
        v5 = 0;
        v6 = 0;
        goto LABEL_5;
      }
      v10 = v20;
      if ( v19 )
      {
        v15 = v20;
        sub_C7D030(&v19);
        sub_C7D280(&v19, v8, v15);
        sub_C7D290(&v19, &v18);
        v10 = v18;
        v9 = v15;
        v14 = 8 * v17;
      }
      v5 = 0;
      v11 = 0;
      v6 = 0;
LABEL_16:
      *(_QWORD *)(*(_QWORD *)(a2 + 272) + v14) = v10;
      goto LABEL_5;
    }
LABEL_8:
    *(_BYTE *)(a1 + 48) |= 1u;
    *(_DWORD *)a1 = v3;
    *(_QWORD *)(a1 + 8) = v4;
    return a1;
  }
  sub_C22080((__int64)&v19, (_QWORD *)a2, &v17);
  if ( (v21 & 1) != 0 )
  {
    v3 = (int)v19;
    v4 = v20;
    if ( (_DWORD)v19 )
      goto LABEL_8;
  }
  v5 = v20;
  v6 = v19;
  v7 = &v19[3 * v20 - 3];
  v8 = (__int64 *)*v7;
  v9 = v7[1];
  v10 = *(_QWORD *)(*(_QWORD *)(a2 + 296) + 8 * v17);
  v11 = 1;
  if ( !v10 )
  {
    v16 = v9;
    v10 = sub_C1B290(v19, &v19[3 * v20]);
    v9 = v16;
    v14 = 8 * v17;
    v11 = 1;
    goto LABEL_16;
  }
LABEL_5:
  *(_BYTE *)(a1 + 48) &= ~1u;
  *(_QWORD *)a1 = v8;
  *(_QWORD *)(a1 + 8) = v9;
  *(_QWORD *)(a1 + 16) = v6;
  *(_QWORD *)(a1 + 24) = v5;
  *(_DWORD *)(a1 + 32) = v11;
  *(_DWORD *)(a1 + 36) = 0;
  *(_QWORD *)(a1 + 40) = v10;
  return a1;
}
