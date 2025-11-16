// Function: sub_735DA0
// Address: 0x735da0
//
__int64 __fastcall sub_735DA0(__int64 a1, int a2, _BYTE *a3)
{
  __int64 v3; // rcx
  __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // rdx
  _BYTE *v7; // rax
  __int64 v8; // [rsp+8h] [rbp-18h] BYREF

  if ( a3 && a2 == -1 )
  {
    v8 = 0;
    v3 = 0;
  }
  else
  {
    v7 = sub_735B90(a2, a1, &v8);
    v3 = v8;
    a3 = v7;
  }
  result = *((_QWORD *)a3 + 14);
  v5 = *(_QWORD *)(a1 + 112);
  if ( a1 != result && result )
  {
    do
    {
      v6 = result;
      result = *(_QWORD *)(result + 112);
    }
    while ( a1 != result && result );
    *(_QWORD *)(v6 + 112) = v5;
  }
  else
  {
    *((_QWORD *)a3 + 14) = v5;
    v6 = 0;
  }
  if ( v3 )
  {
    if ( *(_QWORD *)(v3 + 40) == a1 )
      *(_QWORD *)(v3 + 40) = v6;
  }
  return result;
}
