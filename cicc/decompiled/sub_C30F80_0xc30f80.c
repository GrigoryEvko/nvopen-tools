// Function: sub_C30F80
// Address: 0xc30f80
//
__int64 __fastcall sub_C30F80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  _QWORD v10[3]; // [rsp+8h] [rbp-38h] BYREF
  __int64 v11; // [rsp+20h] [rbp-20h]

  if ( *(_DWORD *)(a1 + 24) == 1 && !*(_BYTE *)(a1 + 296) )
  {
    v8 = *(_QWORD *)a1;
    LOBYTE(v11) = 0;
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, __int64, __int64, __int64, _QWORD, _QWORD, __int64))(v8 + 24))(
      v10,
      a1,
      *(_QWORD *)(a1 + 16),
      a4,
      a5,
      a6,
      v10[1],
      v10[2],
      v11);
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v10[0] + 16LL))(v10[0]);
    v9 = v10[0];
    *(_BYTE *)(a1 + 296) = 1;
    if ( v9 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
  }
  return sub_C30F00(a1, a2);
}
