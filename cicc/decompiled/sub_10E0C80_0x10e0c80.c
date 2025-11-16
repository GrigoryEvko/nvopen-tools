// Function: sub_10E0C80
// Address: 0x10e0c80
//
__int64 __fastcall sub_10E0C80(__int64 a1, unsigned __int64 a2, __int64 **a3, unsigned int a4)
{
  __int64 v5; // r12
  __int64 v6; // rax

  v5 = sub_AD4C30(a2, a3, 0);
  v6 = sub_96F480(a4, v5, *(_QWORD *)(a2 + 8), *(_QWORD *)(a1 + 88));
  if ( v6 && a2 == v6 )
    return v5;
  else
    return 0;
}
