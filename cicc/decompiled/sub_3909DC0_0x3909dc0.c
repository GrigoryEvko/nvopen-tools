// Function: sub_3909DC0
// Address: 0x3909dc0
//
__int64 __fastcall sub_3909DC0(unsigned int *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax

  if ( *(_DWORD *)sub_3909460((__int64)a1) == 9 )
  {
    (*(void (__fastcall **)(unsigned int *))(*(_QWORD *)a1 + 136LL))(a1);
    return 0;
  }
  else
  {
    v3 = sub_3909460((__int64)a1);
    v4 = sub_39092A0(v3);
    return sub_3909790(a1, v4, a2, 0, 0);
  }
}
