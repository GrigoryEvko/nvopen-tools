// Function: sub_1667080
// Address: 0x1667080
//
void __fastcall sub_1667080(__int64 a1, __int64 *a2)
{
  __int64 *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rcx
  char v6; // dl
  char v7; // al
  const char *v8; // rax
  __int64 v9; // r14
  _BYTE *v10; // rax
  __int64 v11; // rax
  const char *v12; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-40h] BYREF
  char v14; // [rsp+10h] [rbp-30h]
  char v15; // [rsp+11h] [rbp-2Fh]

  v3 = (__int64 *)*(a2 - 3);
  v4 = *a2;
  v5 = *v3;
  v6 = *(_BYTE *)(v4 + 8);
  v7 = *(_BYTE *)(*v3 + 8);
  if ( (v7 == 16) == (v6 == 16) )
  {
    if ( v7 == 16 )
    {
      if ( *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL) == 11 )
      {
LABEL_4:
        if ( v6 == 16 )
        {
          if ( (unsigned __int8)(*(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL) - 1) <= 5u )
          {
LABEL_16:
            if ( *(_QWORD *)(v4 + 32) == *(_QWORD *)(v5 + 32) )
              goto LABEL_7;
            v15 = 1;
            v12 = "UIToFP source and dest vector length mismatch";
LABEL_18:
            v13[0] = v12;
            v14 = 3;
            sub_164FF40((__int64 *)a1, (__int64)v13);
            if ( !*(_QWORD *)a1 )
              return;
LABEL_14:
            sub_164FA80((__int64 *)a1, (__int64)a2);
            return;
          }
        }
        else if ( (unsigned __int8)(v6 - 1) <= 5u )
        {
          if ( v7 != 16 )
          {
LABEL_7:
            sub_1663F80(a1, (__int64)a2);
            return;
          }
          goto LABEL_16;
        }
        v15 = 1;
        v12 = "UIToFP result must be FP or FP vector";
        goto LABEL_18;
      }
    }
    else if ( v7 == 11 )
    {
      goto LABEL_4;
    }
    v15 = 1;
    v8 = "UIToFP source must be integer or integer vector";
  }
  else
  {
    v15 = 1;
    v8 = "UIToFP source and dest must both be vector or scalar";
  }
  v9 = *(_QWORD *)a1;
  v13[0] = v8;
  v14 = 3;
  if ( v9 )
  {
    sub_16E2CE0(v13, v9);
    v10 = *(_BYTE **)(v9 + 24);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 16) )
    {
      sub_16E7DE0(v9, 10);
    }
    else
    {
      *(_QWORD *)(v9 + 24) = v10 + 1;
      *v10 = 10;
    }
    v11 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 72) = 1;
    if ( v11 )
      goto LABEL_14;
  }
  else
  {
    *(_BYTE *)(a1 + 72) = 1;
  }
}
