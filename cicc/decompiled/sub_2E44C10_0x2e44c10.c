// Function: sub_2E44C10
// Address: 0x2e44c10
//
__int64 __fastcall sub_2E44C10(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 result; // rax
  __int16 v6; // dx
  __int64 v7; // rdx
  __int64 (__fastcall *v8)(__int64); // rcx

  result = a1;
  v6 = *(_WORD *)(a2 + 68);
  if ( !a4 )
  {
    if ( v6 != 20 )
    {
LABEL_3:
      *(_BYTE *)(a1 + 16) = 0;
      return result;
    }
    goto LABEL_4;
  }
  if ( v6 == 20 )
  {
LABEL_4:
    v7 = *(_QWORD *)(a2 + 32);
    *(_BYTE *)(a1 + 16) = 1;
    *(_QWORD *)a1 = v7;
    *(_QWORD *)(a1 + 8) = v7 + 40;
    return result;
  }
  v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 520LL);
  if ( v8 == sub_2DCA430 )
    goto LABEL_3;
  ((void (__fastcall *)(__int64, __int64, __int64))v8)(a1, a3, a2);
  return a1;
}
