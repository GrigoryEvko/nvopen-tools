// Function: sub_10A9B60
// Address: 0x10a9b60
//
bool __fastcall sub_10A9B60(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  _BYTE *v5; // rax
  _BYTE *v7; // rdx
  __int64 v8; // rax
  _BYTE *v9; // rdi
  __int64 v10; // rdx
  _BYTE *v11; // rdi

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v5 != 47
    || (v10 = *((_QWORD *)v5 - 8)) == 0
    || (**a1 = v10, v11 = (_BYTE *)*((_QWORD *)v5 - 4), *v11 > 0x15u)
    || (*a1[1] = v11, *v11 <= 0x15u) && (*v11 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v11)) )
  {
    v7 = (_BYTE *)*((_QWORD *)a3 - 4);
  }
  else
  {
    v7 = (_BYTE *)*((_QWORD *)a3 - 4);
    result = 1;
    if ( (_BYTE *)*a1[3] == v7 )
      return result;
  }
  result = *v7 == 47
        && (v8 = *((_QWORD *)v7 - 8)) != 0
        && (**a1 = v8, v9 = (_BYTE *)*((_QWORD *)v7 - 4), *v9 <= 0x15u)
        && ((*a1[1] = v9, *v9 > 0x15u) || *v9 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v9))
        && *a1[3] == *((_QWORD *)a3 - 8);
  return result;
}
