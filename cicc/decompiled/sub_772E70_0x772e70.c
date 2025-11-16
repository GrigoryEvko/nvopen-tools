// Function: sub_772E70
// Address: 0x772e70
//
__int64 __fastcall sub_772E70(_QWORD *a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 result; // rax
  _QWORD *v4; // rax
  __int64 v5; // rax

  *(_QWORD *)(a1[1] + 16LL) = *a1;
  v1 = a1[1];
  v2 = *(_QWORD *)(v1 + 8);
  if ( v2 )
  {
    a1[1] = v2;
    result = v2 + 24;
    *a1 = result;
  }
  else
  {
    v4 = (_QWORD *)qword_4F082A0;
    if ( qword_4F082A0 )
    {
      qword_4F082A0 = *(_QWORD *)(qword_4F082A0 + 8);
    }
    else
    {
      v4 = (_QWORD *)sub_823970(0x10000);
      v1 = a1[1];
    }
    *v4 = v1;
    a1[1] = v4;
    v4[1] = 0;
    v5 = a1[1];
    a1[2] = 0;
    result = v5 + 24;
    a1[4] = 0;
    *a1 = result;
  }
  return result;
}
