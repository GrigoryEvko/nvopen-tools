// Function: sub_16673C0
// Address: 0x16673c0
//
void __fastcall sub_16673C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  char v4; // dl
  char v5; // al
  __int64 v6; // r14
  char v7; // cl
  char v8; // al
  unsigned int v9; // ebx
  const char *v10; // rax
  __int64 v11; // r14
  _BYTE *v12; // rax
  __int64 v13; // rax
  const char *v14; // [rsp+0h] [rbp-40h] BYREF
  char v15; // [rsp+10h] [rbp-30h]
  char v16; // [rsp+11h] [rbp-2Fh]

  v3 = **(_QWORD **)(a2 - 24);
  v4 = *(_BYTE *)(v3 + 8);
  v5 = v4;
  if ( v4 == 16 )
    v5 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
  if ( (unsigned __int8)(v5 - 1) > 5u )
  {
    v16 = 1;
    v10 = "FPTrunc only operates on FP";
  }
  else
  {
    v6 = *(_QWORD *)a2;
    v7 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
    v8 = v7;
    if ( v7 == 16 )
      v8 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
    if ( (unsigned __int8)(v8 - 1) > 5u )
    {
      v16 = 1;
      v10 = "FPTrunc only produces an FP";
    }
    else if ( (v7 == 16) == (v4 == 16) )
    {
      v9 = sub_16431D0(v3);
      if ( v9 > (unsigned int)sub_16431D0(v6) )
      {
        sub_1663F80(a1, a2);
        return;
      }
      v16 = 1;
      v10 = "DestTy too big for FPTrunc";
    }
    else
    {
      v16 = 1;
      v10 = "fptrunc source and destination must both be a vector or neither";
    }
  }
  v11 = *(_QWORD *)a1;
  v14 = v10;
  v15 = 3;
  if ( v11 )
  {
    sub_16E2CE0(&v14, v11);
    v12 = *(_BYTE **)(v11 + 24);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 16) )
    {
      sub_16E7DE0(v11, 10);
    }
    else
    {
      *(_QWORD *)(v11 + 24) = v12 + 1;
      *v12 = 10;
    }
    v13 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 72) = 1;
    if ( v13 )
      sub_164FA80((__int64 *)a1, a2);
  }
  else
  {
    *(_BYTE *)(a1 + 72) = 1;
  }
}
