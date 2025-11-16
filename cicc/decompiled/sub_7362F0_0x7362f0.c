// Function: sub_7362F0
// Address: 0x7362f0
//
__int64 __fastcall sub_7362F0(__int64 a1, int a2)
{
  _BYTE *v2; // rax
  _BYTE *v3; // r12
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = sub_735B90(a2, a1, v7);
  v3 = v2;
  if ( !*(_QWORD *)(a1 + 40) )
    sub_72EE40(a1, 0xBu, (__int64)v2);
  result = *((_QWORD *)v3 + 18);
  v5 = v7[0];
  if ( result )
  {
    if ( v7[0] )
    {
      result = *(_QWORD *)(v7[0] + 48);
      *(_QWORD *)(result + 112) = a1;
      *(_QWORD *)(a1 + 112) = 0;
LABEL_6:
      *(_QWORD *)(v5 + 48) = a1;
      return result;
    }
    do
    {
      v6 = result;
      result = *(_QWORD *)(result + 112);
    }
    while ( result );
    *(_QWORD *)(v6 + 112) = a1;
    *(_QWORD *)(a1 + 112) = 0;
  }
  else
  {
    *((_QWORD *)v3 + 18) = a1;
    *(_QWORD *)(a1 + 112) = 0;
    if ( v5 )
      goto LABEL_6;
  }
  return result;
}
