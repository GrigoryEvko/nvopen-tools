// Function: sub_1255430
// Address: 0x1255430
//
unsigned __int64 *__fastcall sub_1255430(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v13[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( !*(_BYTE *)(a2 + 40) )
  {
    v10 = *(_QWORD *)(a2 + 16);
    if ( v10 )
    {
      if ( a3 > (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL))(v10) - *(_QWORD *)(a2 + 24) )
        goto LABEL_15;
      if ( *(_BYTE *)(a2 + 40) )
      {
        v9 = *(_QWORD *)(a2 + 32);
LABEL_3:
        if ( a3 + a4 <= v9 )
        {
LABEL_12:
          (*(void (__fastcall **)(unsigned __int64 *, _QWORD, unsigned __int64, __int64, __int64))(**(_QWORD **)(a2 + 16)
                                                                                                 + 24LL))(
            a1,
            *(_QWORD *)(a2 + 16),
            a3 + *(_QWORD *)(a2 + 24),
            a4,
            a5);
          return a1;
        }
        goto LABEL_4;
      }
      v11 = *(_QWORD *)(a2 + 16);
      if ( v11 )
      {
        if ( a3 + a4 <= (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL))(v11) - *(_QWORD *)(a2 + 24) )
          goto LABEL_12;
        goto LABEL_4;
      }
    }
    else if ( a3 )
    {
      goto LABEL_15;
    }
    if ( !(a3 + a4) )
      goto LABEL_12;
LABEL_4:
    sub_1254FA0(v13, 1);
    goto LABEL_5;
  }
  v9 = *(_QWORD *)(a2 + 32);
  if ( a3 <= v9 )
    goto LABEL_3;
LABEL_15:
  sub_1254FA0(v13, 3);
LABEL_5:
  if ( (v13[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_12;
  *a1 = v13[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  return a1;
}
