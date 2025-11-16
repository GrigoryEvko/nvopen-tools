// Function: sub_22B8FA0
// Address: 0x22b8fa0
//
__int64 __fastcall sub_22B8FA0(__int64 a1)
{
  __int64 result; // rax
  int *v2; // rcx
  __int64 v3; // r8
  int *v4; // r15
  int v5; // edx
  int *v6; // r12
  int v7; // ebx
  int v8; // edx
  int v9; // [rsp-70h] [rbp-70h] BYREF
  int v10; // [rsp-6Ch] [rbp-6Ch] BYREF
  _BYTE v11[104]; // [rsp-68h] [rbp-68h] BYREF

  result = *(unsigned int *)(a1 + 72);
  if ( (_DWORD)result )
  {
    v2 = *(int **)(a1 + 64);
    v3 = 4LL * *(unsigned int *)(a1 + 80);
    v4 = &v2[v3];
    if ( v2 != &v2[v3] )
    {
      while ( 1 )
      {
        v5 = *v2;
        v6 = v2;
        if ( (unsigned int)*v2 <= 0xFFFFFFFD )
          break;
        v2 += 4;
        if ( v4 == v2 )
          return result;
      }
      if ( v2 != v4 )
      {
        v7 = 0;
        while ( 1 )
        {
          v9 = v5;
          v10 = v7;
          v6 += 4;
          sub_22B89D0((__int64)v11, a1 + 88, &v9, &v10);
          v8 = *(v6 - 4);
          v9 = v7++;
          v10 = v8;
          result = sub_22B89D0((__int64)v11, a1 + 120, &v9, &v10);
          if ( v6 == v4 )
            break;
          while ( (unsigned int)*v6 > 0xFFFFFFFD )
          {
            v6 += 4;
            if ( v4 == v6 )
              return result;
          }
          if ( v4 == v6 )
            return result;
          v5 = *v6;
        }
      }
    }
  }
  return result;
}
