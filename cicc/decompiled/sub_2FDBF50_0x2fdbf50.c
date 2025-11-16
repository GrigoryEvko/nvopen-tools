// Function: sub_2FDBF50
// Address: 0x2fdbf50
//
__int64 __fastcall sub_2FDBF50(__int64 a1, __int64 *a2)
{
  __int64 (*v2)(void); // rax
  unsigned int v3; // r12d

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 200LL);
  if ( (char *)v2 == (char *)sub_2E76F30 )
  {
    v3 = sub_B2D610(*a2, 20);
    if ( (_BYTE)v3 )
      return v3;
    v3 = (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 392LL))(a1, a2) ^ 1;
  }
  else
  {
    v3 = v2();
  }
  if ( !(_BYTE)v3 && !(unsigned __int8)sub_B2D610(*a2, 20) )
    return (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 392LL))(a1, a2);
  return v3;
}
