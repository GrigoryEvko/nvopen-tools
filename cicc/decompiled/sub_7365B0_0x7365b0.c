// Function: sub_7365B0
// Address: 0x7365b0
//
__int64 __fastcall sub_7365B0(__int64 a1, int a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rdx
  bool v5; // zf
  __int64 v6; // rdx
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  result = (__int64)sub_735B90(a2, a1, v7);
  if ( !result )
    return result;
  v3 = *(_QWORD *)(result + 104);
  if ( !v3 )
  {
    *(_QWORD *)(result + 104) = a1;
    v5 = *(_QWORD *)(a1 + 40) == 0;
    *(_QWORD *)(a1 + 112) = 0;
    if ( !v5 || (*(_BYTE *)(a1 + 89) & 2) != 0 )
    {
LABEL_15:
      v4 = v7[0];
      if ( !v7[0] )
        goto LABEL_6;
      goto LABEL_5;
    }
    goto LABEL_18;
  }
  v4 = v7[0];
  if ( !v7[0] )
  {
    do
    {
      v6 = v3;
      v3 = *(_QWORD *)(v3 + 112);
    }
    while ( v3 );
    *(_QWORD *)(v6 + 112) = a1;
    v5 = *(_QWORD *)(a1 + 40) == 0;
    *(_QWORD *)(a1 + 112) = 0;
    if ( !v5 || (*(_BYTE *)(a1 + 89) & 2) != 0 )
    {
      v3 = v6;
      goto LABEL_6;
    }
    v3 = v6;
    goto LABEL_18;
  }
  v3 = *(_QWORD *)(v7[0] + 32);
  *(_QWORD *)(v3 + 112) = a1;
  v5 = *(_QWORD *)(a1 + 40) == 0;
  *(_QWORD *)(a1 + 112) = 0;
  if ( v5 && (*(_BYTE *)(a1 + 89) & 2) == 0 )
  {
LABEL_18:
    sub_72EE40(a1, 6u, result);
    goto LABEL_15;
  }
LABEL_5:
  *(_QWORD *)(v4 + 32) = a1;
LABEL_6:
  result = (unsigned int)*(unsigned __int8 *)(a1 + 140) - 9;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
  {
    result = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      result = *(_QWORD *)(result + 96);
      if ( result )
        *(_QWORD *)(result + 160) = v3;
    }
  }
  return result;
}
