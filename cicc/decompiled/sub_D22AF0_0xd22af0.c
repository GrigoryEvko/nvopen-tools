// Function: sub_D22AF0
// Address: 0xd22af0
//
__int64 __fastcall sub_D22AF0(__int64 a1, __int64 *a2, _QWORD *a3, __int64 *a4, _QWORD *a5)
{
  unsigned int v7; // r12d
  __int64 *v9; // rax
  __int64 *v10; // [rsp+0h] [rbp-30h] BYREF
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = sub_D22920(a1, (__int64 *)&v10, v11, a4, a5);
  if ( !(_BYTE)v7 )
    return v7;
  if ( v10 )
  {
    *a2 = *v10;
  }
  else
  {
    v9 = (__int64 *)sub_AA48A0(*a4);
    *a2 = sub_ACD6D0(v9);
  }
  *a3 = *(_QWORD *)v11[0];
  return v7;
}
