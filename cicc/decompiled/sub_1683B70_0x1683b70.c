// Function: sub_1683B70
// Address: 0x1683b70
//
__int64 __fastcall sub_1683B70(_QWORD *a1, void (__fastcall *a2)(__int64))
{
  _QWORD *v2; // rbx
  __int64 result; // rax
  _QWORD *v4; // r12
  __int64 v5; // rdi

  if ( a1 )
  {
    v2 = a1;
    do
    {
      while ( 1 )
      {
        v4 = v2;
        v2 = (_QWORD *)*v2;
        v5 = v4[1];
        if ( !a2 )
          break;
        a2(v5);
        result = sub_16856A0(v4);
        if ( !v2 )
          return result;
      }
      sub_16856A0(v5);
      result = sub_16856A0(v4);
    }
    while ( v2 );
  }
  return result;
}
