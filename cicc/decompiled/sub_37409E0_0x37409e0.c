// Function: sub_37409E0
// Address: 0x37409e0
//
unsigned __int8 *__fastcall sub_37409E0(__int64 *a1, unsigned __int8 *a2)
{
  unsigned __int8 *result; // rax
  __int64 v3; // rax
  unsigned __int8 **v4; // rdx
  unsigned __int8 *v5; // rbx
  _QWORD *v6; // rdx

  result = sub_3247C80((__int64)a1, a2);
  if ( !result )
  {
    v3 = *(a2 - 16);
    if ( (v3 & 2) != 0 )
      v4 = (unsigned __int8 **)*((_QWORD *)a2 - 4);
    else
      v4 = (unsigned __int8 **)&a2[-8 * (((unsigned __int8)v3 >> 2) & 0xF) - 16];
    v5 = sub_373FC60(a1, *v4);
    result = (unsigned __int8 *)sub_3740A90(a1, a2);
    *((_QWORD *)result + 5) = (unsigned __int64)v5 & 0xFFFFFFFFFFFFFFFBLL;
    v6 = (_QWORD *)*((_QWORD *)v5 + 4);
    if ( v6 )
    {
      *(_QWORD *)result = *v6;
      **((_QWORD **)v5 + 4) = (unsigned __int64)result & 0xFFFFFFFFFFFFFFFBLL;
    }
    *((_QWORD *)v5 + 4) = result;
  }
  return result;
}
