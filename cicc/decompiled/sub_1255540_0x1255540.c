// Function: sub_1255540
// Address: 0x1255540
//
unsigned __int64 *__fastcall sub_1255540(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  unsigned __int64 v13; // rax
  __int64 v14[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( !*(_BYTE *)(a2 + 40) )
  {
    v10 = *(_QWORD *)(a2 + 16);
    if ( v10 )
    {
      if ( a3 > (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL))(v10) - *(_QWORD *)(a2 + 24) )
        goto LABEL_21;
      if ( *(_BYTE *)(a2 + 40) )
      {
        v7 = *(_QWORD *)(a2 + 32);
LABEL_3:
        if ( a3 + 1 <= v7 )
          goto LABEL_13;
LABEL_4:
        sub_1254FA0(v14, 1);
        goto LABEL_5;
      }
      v11 = *(_QWORD *)(a2 + 16);
      if ( v11 )
      {
        if ( a3 + 1 <= (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL))(v11) - *(_QWORD *)(a2 + 24) )
          goto LABEL_13;
        goto LABEL_4;
      }
    }
    else if ( a3 )
    {
      goto LABEL_21;
    }
    if ( a3 == -1 )
      goto LABEL_13;
    goto LABEL_4;
  }
  v7 = *(_QWORD *)(a2 + 32);
  if ( a3 <= v7 )
    goto LABEL_3;
LABEL_21:
  sub_1254FA0(v14, 3);
LABEL_5:
  v8 = v14[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v14[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
LABEL_6:
    *a1 = v8 | 1;
    return a1;
  }
LABEL_13:
  (*(void (__fastcall **)(__int64 *, _QWORD, unsigned __int64, __int64))(**(_QWORD **)(a2 + 16) + 32LL))(
    v14,
    *(_QWORD *)(a2 + 16),
    a3 + *(_QWORD *)(a2 + 24),
    a4);
  v8 = v14[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v14[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_6;
  if ( *(_BYTE *)(a2 + 40) )
  {
    v8 = *(_QWORD *)(a2 + 32);
  }
  else
  {
    v12 = *(_QWORD *)(a2 + 16);
    if ( v12 )
      v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 40LL))(v12) - *(_QWORD *)(a2 + 24);
  }
  v13 = v8 - a3;
  if ( v13 < *(_QWORD *)(a4 + 8) )
    *(_QWORD *)(a4 + 8) = v13;
  *a1 = 1;
  return a1;
}
