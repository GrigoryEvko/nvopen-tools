// Function: sub_12558B0
// Address: 0x12558b0
//
unsigned __int64 *__fastcall sub_12558B0(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v12; // rdi
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( ((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 48LL))(*(_QWORD *)(a2 + 16)) & 2) != 0 )
  {
    if ( *(_BYTE *)(a2 + 40) )
    {
      if ( a3 <= *(_QWORD *)(a2 + 32) )
        goto LABEL_11;
    }
    else
    {
      v9 = *(_QWORD *)(a2 + 16);
      v10 = 0;
      if ( v9 )
        v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 40LL))(v9) - *(_QWORD *)(a2 + 24);
      if ( a3 <= v10 )
        goto LABEL_11;
    }
    goto LABEL_6;
  }
  if ( *(_BYTE *)(a2 + 40) )
  {
    v14 = *(_QWORD *)(a2 + 32);
    if ( a3 <= v14 )
      goto LABEL_18;
    goto LABEL_6;
  }
  v12 = *(_QWORD *)(a2 + 16);
  if ( !v12 )
  {
    if ( !a3 )
    {
LABEL_23:
      v14 = 0;
      goto LABEL_18;
    }
LABEL_6:
    sub_1254FA0(v15, 3);
    goto LABEL_7;
  }
  if ( a3 > (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 40LL))(v12) - *(_QWORD *)(a2 + 24) )
    goto LABEL_6;
  if ( *(_BYTE *)(a2 + 40) )
  {
    v14 = *(_QWORD *)(a2 + 32);
    goto LABEL_18;
  }
  v13 = *(_QWORD *)(a2 + 16);
  if ( !v13 )
    goto LABEL_23;
  v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 40LL))(v13) - *(_QWORD *)(a2 + 24);
LABEL_18:
  if ( a3 + a5 <= v14 )
    goto LABEL_11;
  sub_1254FA0(v15, 1);
LABEL_7:
  if ( (v15[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
LABEL_11:
    (*(void (__fastcall **)(unsigned __int64 *, _QWORD, unsigned __int64, __int64, __int64))(**(_QWORD **)(a2 + 16)
                                                                                           + 56LL))(
      a1,
      *(_QWORD *)(a2 + 16),
      a3 + *(_QWORD *)(a2 + 24),
      a4,
      a5);
    return a1;
  }
  *a1 = v15[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  return a1;
}
