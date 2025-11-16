// Function: sub_736270
// Address: 0x736270
//
__int64 __fastcall sub_736270(__int64 a1, int a2)
{
  _BYTE *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  v2 = sub_735B90(a2, a1, &v6);
  v3 = *((_QWORD *)v2 + 18);
  if ( a1 != v3 && v3 )
  {
    do
    {
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 112);
    }
    while ( a1 != v3 && v3 );
    *(_QWORD *)(v4 + 112) = *(_QWORD *)(a1 + 112);
  }
  else
  {
    *((_QWORD *)v2 + 18) = *(_QWORD *)(a1 + 112);
    v4 = 0;
  }
  result = v6;
  if ( v6 )
  {
    if ( *(_QWORD *)(v6 + 48) == a1 )
      *(_QWORD *)(v6 + 48) = v4;
  }
  return result;
}
