// Function: sub_10A4B40
// Address: 0x10a4b40
//
__int64 __fastcall sub_10A4B40(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx
  _BYTE *v6; // rdx

  result = 0;
  if ( *a2 == 82 )
  {
    v3 = sub_B53900((__int64)a2);
    sub_B53630(v3, *(_QWORD *)a1);
    result = v4;
    if ( (_BYTE)v4
      && (v5 = *((_QWORD *)a2 - 8)) != 0
      && (**(_QWORD **)(a1 + 8) = v5, v6 = (_BYTE *)*((_QWORD *)a2 - 4), *v6 == 17) )
    {
      **(_QWORD **)(a1 + 16) = v6;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
