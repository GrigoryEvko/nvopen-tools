// Function: sub_892A30
// Address: 0x892a30
//
__int64 __fastcall sub_892A30(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rbx

  v2 = a1[13];
  if ( !v2 )
    v2 = sub_8790A0((__int64)a1);
  result = sub_883800(*(_QWORD *)(a2 + 96) + 192LL, *a1);
  if ( result )
  {
    while ( *(_BYTE *)(result + 80) != 9 )
    {
      result = *(_QWORD *)(result + 32);
      if ( !result )
        return result;
    }
    v4 = *(_QWORD *)(result + 104);
    if ( *(_DWORD *)v4 == dword_4F06650[0] )
    {
      v5 = *(_QWORD *)(v4 + 8);
      *(_QWORD *)(v2 + 8) = v5;
      if ( v5 )
        *(_QWORD *)(v2 + 16) = result;
    }
    else
    {
      v6 = a1[11];
      result = (__int64)sub_72C9A0();
      *(_QWORD *)(v6 + 184) = result;
    }
  }
  return result;
}
