// Function: sub_38DC390
// Address: 0x38dc390
//
__int64 __fastcall sub_38DC390(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // rdx
  void (*v7)(); // rax
  __int64 v8; // r13
  unsigned __int64 v9; // rdx
  __int64 v10; // rax

  result = *(_QWORD *)(a1 + 112) + 32LL * *(unsigned int *)(a1 + 120) - 32;
  v5 = *(_QWORD *)result;
  v6 = *(_QWORD *)(result + 8);
  *(_QWORD *)(result + 16) = *(_QWORD *)result;
  *(_QWORD *)(result + 24) = v6;
  if ( v6 != a3 || v5 != a2 )
  {
    v7 = *(void (**)())(*(_QWORD *)a1 + 152LL);
    if ( v7 != nullsub_1939 )
      ((void (__fastcall *)(__int64, __int64, __int64))v7)(a1, a2, a3);
    result = *(_QWORD *)(a1 + 112) + 32LL * *(unsigned int *)(a1 + 120) - 32;
    *(_QWORD *)result = a2;
    *(_QWORD *)(result + 8) = a3;
    v8 = *(_QWORD *)(a2 + 8);
    if ( v8 )
    {
      result = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !result )
      {
        if ( (*(_BYTE *)(v8 + 9) & 0xC) != 8 )
          return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 176LL))(a1, v8, 0);
        *(_BYTE *)(v8 + 8) |= 4u;
        v9 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v8 + 24));
        v10 = v9 | *(_QWORD *)v8 & 7LL;
        *(_QWORD *)v8 = v10;
        if ( !v9 )
          return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 176LL))(a1, v8, 0);
        result = v10 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !result )
        {
          result = 0;
          if ( (*(_BYTE *)(v8 + 9) & 0xC) == 8 )
          {
            *(_BYTE *)(v8 + 8) |= 4u;
            result = (__int64)sub_38CE440(*(_QWORD *)(v8 + 24));
            *(_QWORD *)v8 = result | *(_QWORD *)v8 & 7LL;
          }
        }
      }
      if ( off_4CF6DB8 != (_UNKNOWN *)result )
        return result;
      return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 176LL))(a1, v8, 0);
    }
  }
  return result;
}
