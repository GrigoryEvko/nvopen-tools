// Function: sub_38D7240
// Address: 0x38d7240
//
__int64 __fastcall sub_38D7240(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  unsigned int v4; // ecx
  int *v5; // rdx
  int v6; // r8d
  int v8; // edx
  int v9; // r10d

  v2 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 168);
    v4 = (v2 - 1) & (37 * a2);
    v5 = (int *)(v3 + 8LL * v4);
    v6 = *v5;
    if ( *v5 == a2 )
    {
LABEL_3:
      if ( v5 != (int *)(v3 + 8 * v2) )
        return (unsigned int)v5[1];
    }
    else
    {
      v8 = 1;
      while ( v6 != -1 )
      {
        v9 = v8 + 1;
        v4 = (v2 - 1) & (v8 + v4);
        v5 = (int *)(v3 + 8LL * v4);
        v6 = *v5;
        if ( *v5 == a2 )
          goto LABEL_3;
        v8 = v9;
      }
    }
  }
  return a2;
}
