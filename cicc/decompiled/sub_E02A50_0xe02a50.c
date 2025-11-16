// Function: sub_E02A50
// Address: 0xe02a50
//
unsigned __int8 *__fastcall sub_E02A50(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int8 *v3; // rax
  unsigned __int8 *v4; // rdx
  unsigned __int8 *result; // rax

  v3 = (unsigned __int8 *)sub_E027A0(*(_QWORD *)(a1 - 32), a2, a3, a1);
  if ( !v3 )
    return 0;
  v4 = sub_BD3990(v3, a2);
  if ( !*v4 )
    return v4;
  if ( *v4 != 1 )
    return 0;
  result = (unsigned __int8 *)*((_QWORD *)v4 - 4);
  if ( *result )
    return 0;
  return result;
}
