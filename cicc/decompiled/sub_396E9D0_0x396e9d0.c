// Function: sub_396E9D0
// Address: 0x396e9d0
//
__int64 __fastcall sub_396E9D0(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 (__fastcall *v7)(__int64, __int64, __int64); // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  bool v10; // zf

  result = a2[32] & 0xF;
  if ( (unsigned __int8)result <= 5u )
  {
    if ( (unsigned __int8)result <= 1u )
    {
      v6 = *(_QWORD *)(a1 + 256);
      v7 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v6 + 256LL);
LABEL_6:
      v8 = 8;
      return v7(v6, a3, v8);
    }
  }
  else if ( (unsigned __int8)result <= 8u )
  {
    return result;
  }
  v9 = *(_QWORD *)(a1 + 240);
  v6 = *(_QWORD *)(a1 + 256);
  if ( !*(_BYTE *)(v9 + 328) )
  {
    v10 = *(_BYTE *)(v9 + 330) == 0;
    v8 = 20;
    v7 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v6 + 256LL);
    if ( v10 )
      return v7(v6, a3, v8);
    goto LABEL_6;
  }
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v6 + 256LL))(v6, a3, 8);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 329LL) && (unsigned __int8)sub_15E5030(a2) )
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 256) + 256LL))(
             *(_QWORD *)(a1 + 256),
             a3,
             23);
  else
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 256) + 256LL))(
             *(_QWORD *)(a1 + 256),
             a3,
             21);
}
