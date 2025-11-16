// Function: sub_16668F0
// Address: 0x16668f0
//
void __fastcall sub_16668F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  char v4; // al
  char v5; // dl
  __int64 v6; // r14
  char v7; // dl
  unsigned int v8; // ebx
  const char *v9; // rax
  __int64 v10; // r14
  _BYTE *v11; // rax
  __int64 v12; // rax
  const char *v13; // [rsp+0h] [rbp-40h] BYREF
  char v14; // [rsp+10h] [rbp-30h]
  char v15; // [rsp+11h] [rbp-2Fh]

  v3 = **(_QWORD **)(a2 - 24);
  v4 = *(_BYTE *)(v3 + 8);
  v5 = v4;
  if ( v4 == 16 )
    v5 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
  if ( v5 != 11 )
  {
    v15 = 1;
    v9 = "Trunc only operates on integer";
    goto LABEL_11;
  }
  v6 = *(_QWORD *)a2;
  v7 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v7 != 16 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 11 )
      goto LABEL_6;
LABEL_10:
    v15 = 1;
    v9 = "Trunc only produces integer";
    goto LABEL_11;
  }
  if ( *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL) != 11 )
    goto LABEL_10;
LABEL_6:
  if ( (v7 == 16) == (v4 == 16) )
  {
    v8 = sub_16431D0(v3);
    if ( v8 > (unsigned int)sub_16431D0(v6) )
    {
      sub_1663F80(a1, a2);
      return;
    }
    v15 = 1;
    v9 = "DestTy too big for Trunc";
  }
  else
  {
    v15 = 1;
    v9 = "trunc source and destination must both be a vector or neither";
  }
LABEL_11:
  v10 = *(_QWORD *)a1;
  v13 = v9;
  v14 = 3;
  if ( v10 )
  {
    sub_16E2CE0(&v13, v10);
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
      sub_164FA80((__int64 *)a1, a2);
  }
  else
  {
    *(_BYTE *)(a1 + 72) = 1;
  }
}
