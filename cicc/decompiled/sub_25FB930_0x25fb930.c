// Function: sub_25FB930
// Address: 0x25fb930
//
__int64 __fastcall sub_25FB930(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // edx
  int *v5; // rcx
  int v6; // edi
  int v8; // ecx
  int v9; // r10d
  __int64 v10; // [rsp+0h] [rbp-8h]

  v2 = *(unsigned int *)(a1 + 112);
  v3 = *(_QWORD *)(a1 + 96);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (37 * a2);
    v5 = (int *)(v3 + 8LL * v4);
    v6 = *v5;
    if ( *v5 == a2 )
    {
LABEL_3:
      if ( v5 != (int *)(v3 + 8 * v2) )
      {
        BYTE4(v10) = 1;
        LODWORD(v10) = v5[1];
        return v10;
      }
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
  BYTE4(v10) = 0;
  return v10;
}
