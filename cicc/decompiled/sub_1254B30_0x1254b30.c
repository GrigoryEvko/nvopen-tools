// Function: sub_1254B30
// Address: 0x1254b30
//
__int64 *__fastcall sub_1254B30(__int64 *a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v8; // rdi
  __int64 v9; // rax

  if ( *(_BYTE *)(a2 + 48) )
  {
    v5 = *(_QWORD *)(a2 + 40);
  }
  else
  {
    v8 = *(_QWORD *)(a2 + 24);
    v5 = 0;
    if ( v8 )
    {
      v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 40LL))(v8);
      v6 = *(_QWORD *)(a2 + 56);
      if ( a3 > v9 - *(_QWORD *)(a2 + 32) - v6 )
        goto LABEL_4;
      goto LABEL_7;
    }
  }
  v6 = *(_QWORD *)(a2 + 56);
  if ( a3 > v5 - v6 )
  {
LABEL_4:
    sub_12547E0(a1, 1);
    return a1;
  }
LABEL_7:
  *(_QWORD *)(a2 + 56) = v6 + a3;
  *a1 = 1;
  return a1;
}
