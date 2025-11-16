// Function: sub_1666D40
// Address: 0x1666d40
//
void __fastcall sub_1666D40(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  char v6; // cl
  char v7; // al
  char v8; // dl
  const char *v9; // rax
  __int64 v10; // r14
  _BYTE *v11; // rax
  __int64 v12; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-40h] BYREF
  char v14; // [rsp+10h] [rbp-30h]
  char v15; // [rsp+11h] [rbp-2Fh]

  v4 = *a2;
  v5 = *(_QWORD *)*(a2 - 3);
  v6 = *(_BYTE *)(v4 + 8);
  v7 = *(_BYTE *)(v5 + 8);
  if ( (v7 == 16) == (v6 == 16) )
  {
    v8 = *(_BYTE *)(v5 + 8);
    if ( v7 == 16 )
      v8 = *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL);
    if ( (unsigned __int8)(v8 - 1) > 5u )
    {
      v15 = 1;
      v9 = "FPToUI source must be FP or FP vector";
    }
    else
    {
      if ( v6 == 16 )
      {
        if ( *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL) == 11 )
          goto LABEL_8;
      }
      else if ( v6 == 11 )
      {
        if ( v7 != 16 )
        {
LABEL_9:
          sub_1663F80(a1, (__int64)a2);
          return;
        }
LABEL_8:
        if ( *(_QWORD *)(v4 + 32) == *(_QWORD *)(v5 + 32) )
          goto LABEL_9;
        v15 = 1;
        v13[0] = "FPToUI source and dest vector length mismatch";
        v14 = 3;
        sub_164FF40((__int64 *)a1, (__int64)v13);
        if ( !*(_QWORD *)a1 )
          return;
LABEL_15:
        sub_164FA80((__int64 *)a1, (__int64)a2);
        return;
      }
      v15 = 1;
      v9 = "FPToUI result must be integer or integer vector";
    }
  }
  else
  {
    v15 = 1;
    v9 = "FPToUI source and dest must both be vector or scalar";
  }
  v10 = *(_QWORD *)a1;
  v13[0] = v9;
  v14 = 3;
  if ( !v10 )
  {
    *(_BYTE *)(a1 + 72) = 1;
    return;
  }
  sub_16E2CE0(v13, v10);
  v11 = *(_BYTE **)(v10 + 24);
  if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 16) )
  {
    sub_16E7DE0(v10, 10);
  }
  else
  {
    *(_QWORD *)(v10 + 24) = v11 + 1;
    *v11 = 10;
  }
  v12 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 72) = 1;
  if ( v12 )
    goto LABEL_15;
}
