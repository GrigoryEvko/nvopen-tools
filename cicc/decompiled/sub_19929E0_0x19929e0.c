// Function: sub_19929E0
// Address: 0x19929e0
//
__int64 __fastcall sub_19929E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 i; // r14
  __int64 v5; // r15
  __int64 v6; // rbx
  __int64 v7; // rax

  for ( i = *(_QWORD *)(a2 + 24); a2 + 8 != i; i = sub_220EF30(i) )
  {
    v5 = *(_QWORD *)(i + 32);
    if ( v5 == a1 )
      break;
    v6 = sub_1456040(a1);
    if ( v6 == sub_1456040(v5) )
    {
      v7 = sub_1C551F0(a1, v5, a3);
      if ( v7 )
      {
        *a4 = v7;
        return v5;
      }
    }
  }
  return 0;
}
