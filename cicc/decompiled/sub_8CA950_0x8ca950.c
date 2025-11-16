// Function: sub_8CA950
// Address: 0x8ca950
//
__int64 __fastcall sub_8CA950(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  _QWORD *v3; // rdx
  _QWORD *v4; // r12
  _QWORD *i; // rbx

  result = sub_8CA280(a1);
  if ( a1 == result )
  {
    for ( i = (_QWORD *)a2[7]; i; i = (_QWORD *)*i )
      result = sub_8CA0A0(i[2], 1u);
  }
  else
  {
    result = **(_QWORD **)(*(_QWORD *)(result + 152) + 168LL);
    v3 = **(_QWORD ***)(*(_QWORD *)(a1 + 152) + 168LL);
    if ( result && v3 != a2 )
    {
      do
      {
        v3 = (_QWORD *)*v3;
        result = *(_QWORD *)result;
      }
      while ( a2 != v3 && result );
    }
    v4 = (_QWORD *)a2[7];
    if ( result )
    {
      return sub_8CA8D0((_QWORD *)a2[7], *(_QWORD **)(result + 56));
    }
    else
    {
      while ( v4 )
      {
        result = sub_8CA0A0(v4[2], 1u);
        v4 = (_QWORD *)*v4;
      }
    }
  }
  return result;
}
