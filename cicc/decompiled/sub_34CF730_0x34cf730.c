// Function: sub_34CF730
// Address: 0x34cf730
//
__int64 __fastcall sub_34CF730(_QWORD *a1, __int64 a2)
{
  char v3; // al
  int v4; // eax
  unsigned int v5; // r12d
  __int64 v6; // rax
  __int64 (__fastcall *v7)(_QWORD *, __int64 *); // rax
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v11; // [rsp+8h] [rbp-38h] BYREF
  __int64 v12; // [rsp+10h] [rbp-30h] BYREF
  char v13; // [rsp+18h] [rbp-28h]

  (*(void (__fastcall **)(__int64 *, _QWORD *))(*a1 + 216LL))(&v12, a1);
  v3 = v13;
  v13 &= ~2u;
  v4 = v3 & 1;
  if ( !v4 )
  {
    v7 = *(__int64 (__fastcall **)(_QWORD *, __int64 *))(a1[1] + 120LL);
    if ( !v7 )
    {
      v9 = v12;
      v5 = 1;
      goto LABEL_9;
    }
LABEL_5:
    v8 = v7(a1, &v12);
    if ( v8 )
    {
      v5 = 0;
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 16LL))(a2, v8, 1);
    }
    else
    {
      v5 = 1;
    }
    if ( (v13 & 2) != 0 )
      sub_34CF6C0(&v12, v8);
    v9 = v12;
    if ( (v13 & 1) != 0 )
    {
      if ( v12 )
        (*(void (**)(void))(*(_QWORD *)v12 + 8LL))();
      return v5;
    }
LABEL_9:
    if ( v9 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 56LL))(v9);
    return v5;
  }
  v5 = v4;
  v6 = v12;
  v12 = 0;
  v11 = v6 | 1;
  if ( (v6 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v11, (__int64)a1);
  v7 = *(__int64 (__fastcall **)(_QWORD *, __int64 *))(a1[1] + 120LL);
  if ( v7 )
    goto LABEL_5;
  return v5;
}
