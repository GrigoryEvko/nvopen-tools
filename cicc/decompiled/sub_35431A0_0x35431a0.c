// Function: sub_35431A0
// Address: 0x35431a0
//
__int64 __fastcall sub_35431A0(__int64 a1, __int64 a2)
{
  _DWORD *v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rcx
  unsigned int v5; // edx

  v2 = *(_DWORD **)a2;
  v3 = 0;
  v4 = *(_QWORD *)a2 + 88LL * *(unsigned int *)(a2 + 8);
  if ( v4 != *(_QWORD *)a2 )
  {
    do
    {
      if ( v2[10] )
      {
        v5 = v2[20];
        v2[13] = v5;
        if ( v3 < v5 )
          v3 = v5;
      }
      v2 += 22;
    }
    while ( v2 != (_DWORD *)v4 );
  }
  return v3;
}
