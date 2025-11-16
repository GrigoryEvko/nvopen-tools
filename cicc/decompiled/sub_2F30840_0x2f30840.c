// Function: sub_2F30840
// Address: 0x2f30840
//
__int64 __fastcall sub_2F30840(_QWORD *a1, __int64 *a2)
{
  unsigned int v2; // eax
  __int64 v3; // r14
  unsigned int v4; // r13d
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  _QWORD v16[7]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v17; // [rsp+38h] [rbp-38h]
  __int64 v18; // [rsp+40h] [rbp-30h]
  unsigned int v19; // [rsp+48h] [rbp-28h]

  v2 = sub_BB98D0(a1, *a2);
  if ( (_BYTE)v2 )
    return 0;
  v3 = 0;
  v4 = v2;
  if ( (_BYTE)qword_5022BE8 )
  {
    v13 = (__int64 *)a1[1];
    v14 = *v13;
    v15 = v13[1];
    if ( v15 == v14 )
LABEL_19:
      BUG();
    while ( *(_UNKNOWN **)v14 != &unk_501FE44 )
    {
      v14 += 16;
      if ( v15 == v14 )
        goto LABEL_19;
    }
    v3 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(
           *(_QWORD *)(v14 + 8),
           &unk_501FE44)
       + 200;
  }
  v5 = (__int64 *)a1[1];
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_50208AC )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_20;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_50208AC);
  v16[0] = off_4A2A708;
  memset(&v16[1], 0, 24);
  v16[4] = v3;
  v16[5] = v8 + 200;
  v16[6] = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  if ( (_BYTE)qword_5022B08 )
  {
    v10 = 0;
    v11 = 0;
  }
  else
  {
    v9 = sub_2F2D9F0(v16, (__int64)a2);
    v10 = v17;
    v4 = v9;
    v11 = 16LL * v19;
  }
  v16[0] = off_4A2A708;
  sub_C7D6A0(v10, v11, 8);
  return v4;
}
