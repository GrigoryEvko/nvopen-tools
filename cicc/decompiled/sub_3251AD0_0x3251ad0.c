// Function: sub_3251AD0
// Address: 0x3251ad0
//
unsigned __int8 *__fastcall sub_3251AD0(__int64 *a1, unsigned __int8 *a2)
{
  unsigned __int8 *v2; // rbx
  unsigned __int8 v3; // al
  unsigned __int8 *v4; // rbx
  unsigned __int8 *v5; // r15
  __int64 v6; // r14
  unsigned __int8 *result; // rax
  unsigned __int8 v8; // al
  unsigned __int8 *v9; // rbx
  __int64 *v10; // rax

  if ( !a2 )
    return 0;
  v2 = a2 - 16;
  if ( (unsigned __int16)sub_AF18C0((__int64)a2) == 55 && (unsigned __int16)sub_3220AA0(a1[26]) <= 2u
    || (unsigned __int16)sub_AF18C0((__int64)a2) == 71 && (unsigned __int16)sub_3220AA0(a1[26]) <= 4u )
  {
    v8 = *(a2 - 16);
    if ( (v8 & 2) != 0 )
      v9 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    else
      v9 = &v2[-8 * ((v8 >> 2) & 0xF)];
    return (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64 *, _QWORD))(*a1 + 40))(a1, *((_QWORD *)v9 + 3));
  }
  else
  {
    v3 = *(a2 - 16);
    if ( (v3 & 2) != 0 )
      v4 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    else
      v4 = &v2[-8 * ((v3 >> 2) & 0xF)];
    v5 = (unsigned __int8 *)*((_QWORD *)v4 + 1);
    v6 = (*(__int64 (__fastcall **)(__int64 *, unsigned __int8 *))(*a1 + 48))(a1, v5);
    result = sub_3247C80((__int64)a1, a2);
    if ( !result )
    {
      if ( v6 && sub_3215100(v6) )
      {
        v10 = (__int64 *)sub_3215100(v6);
        return (unsigned __int8 *)sub_3251820(v10, v5, v6, a2);
      }
      else
      {
        return (unsigned __int8 *)sub_3251820(a1, v5, v6, a2);
      }
    }
  }
  return result;
}
