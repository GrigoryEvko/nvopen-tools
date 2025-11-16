// Function: sub_28C4AB0
// Address: 0x28c4ab0
//
unsigned __int8 *__fastcall sub_28C4AB0(__int64 *a1, unsigned __int8 *a2, __int64 **a3)
{
  int v4; // edx
  unsigned __int8 *result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9

  if ( !sub_D97040(a1[3], *((_QWORD *)a2 + 1)) )
    return 0;
  v4 = *a2;
  switch ( v4 )
  {
    case '.':
      goto LABEL_8;
    case '?':
      *a3 = sub_DD8400(a1[3], (__int64)a2);
      return sub_28C4120(a1, (__int64)a2, v6, v7, v8, v9);
    case '*':
LABEL_8:
      *a3 = sub_DD8400(a1[3], (__int64)a2);
      return (unsigned __int8 *)sub_28C26C0((__int64)a1, (__int64)a2);
    default:
      result = 0;
      if ( *(_BYTE *)(*((_QWORD *)a2 + 1) + 8LL) == 12 )
      {
        result = sub_28C4470((__int64)a1, (__int64)a2, a3);
        if ( !result )
        {
          result = sub_28C4600((__int64)a1, (__int64)a2, a3);
          if ( !result )
          {
            result = sub_28C4790((__int64)a1, (__int64)a2, a3);
            if ( !result )
              return sub_28C4920((__int64)a1, (__int64)a2, a3);
          }
        }
      }
      break;
  }
  return result;
}
