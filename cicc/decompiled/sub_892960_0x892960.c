// Function: sub_892960
// Address: 0x892960
//
__int64 __fastcall sub_892960(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 i; // rax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  _QWORD *v7; // rax
  _QWORD *v8; // rbx

  v2 = a1[13];
  for ( i = *(_QWORD *)(a2 + 88);
        *(_BYTE *)(*(_QWORD *)(i + 168) + 113LL) || (*(_BYTE *)(i + 177) & 8) != 0;
        i = *(_QWORD *)(*(_QWORD *)(i + 40) + 32LL) )
  {
    ;
  }
  for ( ; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = sub_883800(*(_QWORD *)(*(_QWORD *)i + 96LL) + 192LL, *a1);
  if ( result )
  {
    while ( *(_BYTE *)(result + 80) != 8 )
    {
      result = *(_QWORD *)(result + 32);
      if ( !result )
        return result;
    }
    result = *(_QWORD *)(result + 104);
    if ( *(_DWORD *)result == dword_4F06650[0] )
    {
      v5 = *(_QWORD *)(result + 8);
      *(_QWORD *)(v2 + 16) = result;
      *(_QWORD *)(v2 + 8) = v5;
    }
    else
    {
      v6 = a1[11];
      v7 = sub_725A70(2u);
      *(_QWORD *)(v6 + 152) = v7;
      v8 = v7;
      result = (__int64)sub_72C9A0();
      v8[7] = result;
    }
  }
  return result;
}
