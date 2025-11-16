// Function: sub_174A310
// Address: 0x174a310
//
__int64 __fastcall sub_174A310(__int64 a1)
{
  __int64 v1; // rbx
  _QWORD *v2; // rax
  __int16 *v3; // rax
  __int16 *v4; // rax
  __int16 *v5; // rax
  _QWORD *v6; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax

  v1 = *(_QWORD *)a1;
  v2 = (_QWORD *)sub_16498A0(a1);
  if ( v1 == sub_1643300(v2) )
    return 0;
  v3 = (__int16 *)sub_1698260();
  if ( sub_174A230(a1, v3) )
  {
    v8 = (_QWORD *)sub_16498A0(a1);
    return sub_1643290(v8);
  }
  v4 = (__int16 *)sub_1698270();
  if ( !sub_174A230(a1, v4) )
  {
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 3 )
    {
      v5 = (__int16 *)sub_1698280();
      if ( sub_174A230(a1, v5) )
      {
        v6 = (_QWORD *)sub_16498A0(a1);
        return sub_16432B0(v6);
      }
    }
    return 0;
  }
  v9 = (_QWORD *)sub_16498A0(a1);
  return sub_16432A0(v9);
}
