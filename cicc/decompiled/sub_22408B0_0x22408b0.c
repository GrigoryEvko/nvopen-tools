// Function: sub_22408B0
// Address: 0x22408b0
//
__int64 __fastcall sub_22408B0(_QWORD *a1)
{
  unsigned __int8 *v1; // rax
  unsigned int v2; // r8d
  __int64 (*v4)(void); // rdx
  __int64 (*v5)(); // rax
  unsigned __int8 *v6; // rax
  unsigned int v7; // r8d

  v1 = (unsigned __int8 *)a1[2];
  if ( (unsigned __int64)v1 < a1[3] )
  {
    v2 = *v1;
    a1[2] = v1 + 1;
    return v2;
  }
  v4 = *(__int64 (**)(void))(*a1 + 80LL);
  if ( (char *)v4 != (char *)sub_2240650 )
    return v4();
  v5 = *(__int64 (**)())(*a1 + 72LL);
  if ( v5 == sub_2240390 )
    return (unsigned int)-1;
  if ( (unsigned int)v5() == -1 )
  {
    return (unsigned int)-1;
  }
  else
  {
    v6 = (unsigned __int8 *)a1[2];
    v7 = *v6;
    a1[2] = v6 + 1;
  }
  return v7;
}
