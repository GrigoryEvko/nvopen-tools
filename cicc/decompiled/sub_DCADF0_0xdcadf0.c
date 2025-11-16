// Function: sub_DCADF0
// Address: 0xdcadf0
//
_QWORD *__fastcall sub_DCADF0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax

  if ( *(_BYTE *)a2 == 62 )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
LABEL_3:
    v3 = (__int64 *)sub_BD5C60(a2);
    v4 = sub_BCE3C0(v3, 0);
    v5 = sub_D97090((__int64)a1, v4);
    return sub_DCAD70(a1, v5, v2);
  }
  if ( *(_BYTE *)a2 == 61 )
  {
    v2 = *(_QWORD *)(a2 + 8);
    goto LABEL_3;
  }
  return 0;
}
