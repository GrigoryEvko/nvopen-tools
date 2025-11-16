// Function: sub_3590A60
// Address: 0x3590a60
//
__int64 __fastcall sub_3590A60(_QWORD **a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v4; // r14d
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rax

  v2 = *(_QWORD *)(a2 + 24);
  if ( v2 == a2 + 8 )
  {
    return 0;
  }
  else
  {
    v4 = 0;
    do
    {
      v6 = *a1;
      if ( !(_BYTE)v4 )
      {
        v7 = *(unsigned int *)(v2 + 32);
        if ( (int)v7 >= 0 )
          v5 = *(_QWORD *)(v6[38] + 8 * v7);
        else
          v5 = *(_QWORD *)(v6[7] + 16 * (v7 & 0x7FFFFFFF) + 8);
        LOBYTE(v4) = v5 != 0;
      }
      sub_2EBECB0(v6, *(_DWORD *)(v2 + 32), *(_DWORD *)(v2 + 36));
      v2 = sub_220EF30(v2);
    }
    while ( a2 + 8 != v2 );
  }
  return v4;
}
