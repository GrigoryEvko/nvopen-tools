// Function: sub_F26140
// Address: 0xf26140
//
__int64 __fastcall sub_F26140(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // rsi
  __int64 v9[8]; // [rsp+10h] [rbp-40h] BYREF

  for ( result = *(unsigned int *)(a2 + 8); (_DWORD)result; result = *(unsigned int *)(a2 + 8) )
  {
    v3 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * (unsigned int)result - 8);
    *(_DWORD *)(a2 + 8) = result - 1;
    v4 = *(_QWORD *)(v3 + 16);
    if ( v4 )
    {
      while ( 1 )
      {
        v5 = *(_QWORD *)(v4 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v5 - 30) <= 0xAu )
          break;
        v4 = *(_QWORD *)(v4 + 8);
        if ( !v4 )
          goto LABEL_8;
      }
LABEL_6:
      v6 = *(_QWORD *)(v5 + 40);
      v9[1] = v3;
      v9[0] = v6;
      if ( !sub_F11A70(a1 + 248, v9) && !(unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 80), v3, v6) )
        continue;
      while ( 1 )
      {
        v4 = *(_QWORD *)(v4 + 8);
        if ( !v4 )
          break;
        v5 = *(_QWORD *)(v4 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v5 - 30) <= 0xAu )
          goto LABEL_6;
      }
    }
LABEL_8:
    v7 = *(_QWORD *)(v3 + 56);
    if ( v7 )
      v7 -= 24;
    sub_F25EE0(a1, v7, a2);
  }
  return result;
}
