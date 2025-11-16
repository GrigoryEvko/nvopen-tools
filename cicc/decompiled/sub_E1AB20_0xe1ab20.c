// Function: sub_E1AB20
// Address: 0xe1ab20
//
__int64 __fastcall sub_E1AB20(_QWORD *a1)
{
  _BYTE *v1; // rax
  _BYTE *v2; // rdx
  char v4; // dl
  __int64 v5; // rdx
  _BYTE *v6; // rax

  v1 = (_BYTE *)*a1;
  v2 = (_BYTE *)a1[1];
  if ( (_BYTE *)*a1 == v2 )
    return 0;
  if ( *v1 != 68 )
    return 0;
  *a1 = v1 + 1;
  if ( v2 == v1 + 1 )
    return 0;
  v4 = v1[1];
  if ( v4 != 116 && v4 != 84 )
    return 0;
  *a1 = v1 + 2;
  v5 = sub_E18BB0((__int64)a1);
  if ( !v5 )
    return 0;
  v6 = (_BYTE *)*a1;
  if ( *a1 == a1[1] || *v6 != 69 )
    return 0;
  *a1 = v6 + 1;
  return sub_E0FF70((__int64)(a1 + 102), "decltype", v5);
}
