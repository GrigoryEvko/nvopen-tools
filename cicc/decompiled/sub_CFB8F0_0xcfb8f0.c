// Function: sub_CFB8F0
// Address: 0xcfb8f0
//
__int64 __fastcall sub_CFB8F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  unsigned int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // r8
  int v8; // edx
  int v9; // r10d

  v2 = *(unsigned int *)(a1 + 200);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 184);
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = v3 + 48LL * v4;
    v6 = *(_QWORD *)(v5 + 24);
    if ( a2 == v6 )
    {
LABEL_3:
      if ( v5 != v3 + 48 * v2 )
        return *(_QWORD *)(v5 + 40);
    }
    else
    {
      v8 = 1;
      while ( v6 != -4096 )
      {
        v9 = v8 + 1;
        v4 = (v2 - 1) & (v8 + v4);
        v5 = v3 + 48LL * v4;
        v6 = *(_QWORD *)(v5 + 24);
        if ( a2 == v6 )
          goto LABEL_3;
        v8 = v9;
      }
    }
  }
  return 0;
}
