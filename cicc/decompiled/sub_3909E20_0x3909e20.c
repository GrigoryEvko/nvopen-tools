// Function: sub_3909E20
// Address: 0x3909e20
//
__int64 __fastcall sub_3909E20(unsigned int *a1, int a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rax

  if ( a2 == 9 )
    return sub_3909DC0(a1, a3);
  if ( a2 == *(_DWORD *)sub_3909460((__int64)a1) )
  {
    (*(void (__fastcall **)(unsigned int *))(*(_QWORD *)a1 + 136LL))(a1);
    return 0;
  }
  else
  {
    v5 = sub_3909460((__int64)a1);
    v6 = sub_39092A0(v5);
    return sub_3909790(a1, v6, a3, 0, 0);
  }
}
