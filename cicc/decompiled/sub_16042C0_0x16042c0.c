// Function: sub_16042C0
// Address: 0x16042c0
//
__int64 __fastcall sub_16042C0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r14
  __int64 v3; // r15
  __int64 v4; // rbx

  v1 = *(_QWORD *)(a1 + 32);
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 32);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 32);
      if ( v3 )
      {
        v4 = *(_QWORD *)(v3 + 32);
        if ( v4 )
        {
          sub_16042C0(*(_QWORD *)(v3 + 32));
          sub_1648B90(v4);
        }
        sub_164BE60(v3);
        sub_1648B90(v3);
      }
      sub_164BE60(v2);
      sub_1648B90(v2);
    }
    sub_164BE60(v1);
    sub_1648B90(v1);
  }
  return sub_164BE60(a1);
}
