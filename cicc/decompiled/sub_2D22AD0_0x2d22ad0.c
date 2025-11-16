// Function: sub_2D22AD0
// Address: 0x2d22ad0
//
__int64 __fastcall sub_2D22AD0(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // ecx
  int *v5; // rdx
  int v6; // edi
  int v8; // edx
  int v9; // r10d

  v2 = *(unsigned int *)(a1 + 32);
  v3 = *(_QWORD *)(a1 + 16);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (37 * a2);
    v5 = (int *)(v3 + 72LL * v4);
    v6 = *v5;
    if ( *v5 == a2 )
    {
LABEL_3:
      if ( v5 != (int *)(v3 + 72 * v2) )
        return *((_QWORD *)v5 + 1);
    }
    else
    {
      v8 = 1;
      while ( v6 != -1 )
      {
        v9 = v8 + 1;
        v4 = (v2 - 1) & (v8 + v4);
        v5 = (int *)(v3 + 72LL * v4);
        v6 = *v5;
        if ( *v5 == a2 )
          goto LABEL_3;
        v8 = v9;
      }
    }
  }
  return 0;
}
