// Function: sub_B144A0
// Address: 0xb144a0
//
__int64 __fastcall sub_B144A0(__int64 *a1)
{
  _QWORD *v2; // rbx
  __int64 v3; // rdi
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v6; // rax

  v2 = (_QWORD *)*a1;
  if ( (__int64 *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL) == a1 + 1 )
  {
    result = sub_B14200(a1);
    v2[8] = 0;
  }
  else
  {
    v3 = sub_AA6190(v2[5], (__int64)v2);
    if ( v3 )
    {
      sub_B14410(v3, (__int64)a1, 1);
      result = sub_B14200(a1);
    }
    else
    {
      v5 = v2[4];
      result = sub_B14180((__int64)a1) + 48;
      if ( v5 == result )
      {
        v6 = sub_B14180((__int64)a1);
        result = (__int64)sub_AA7790(v6, (__int64)a1);
        *a1 = 0;
      }
      else
      {
        if ( !v5 )
        {
          MEMORY[0x40] = a1;
          BUG();
        }
        *(_QWORD *)(v5 + 40) = a1;
        *a1 = v5 - 24;
      }
    }
    v2[8] = 0;
  }
  return result;
}
