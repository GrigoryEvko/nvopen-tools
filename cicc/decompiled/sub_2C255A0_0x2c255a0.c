// Function: sub_2C255A0
// Address: 0x2c255a0
//
__int64 __fastcall sub_2C255A0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx

  v2 = a1[1];
  if ( v2 && a2 == v2 + 40 )
    return 0;
  else
    return (*(unsigned int (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a2 + 16LL))(a2, *a1) ^ 1;
}
