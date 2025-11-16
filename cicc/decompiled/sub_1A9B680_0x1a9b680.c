// Function: sub_1A9B680
// Address: 0x1a9b680
//
__int64 __fastcall sub_1A9B680(__int64 a1, __int64 a2)
{
  __int64 *v3; // r12
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 *v8; // rcx
  __int64 v9; // r9
  __int64 v10; // rdx
  int v11; // ecx
  int v12; // r10d
  unsigned __int64 v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v13[0] = a1;
  v3 = (__int64 *)sub_177C990(a2, v13);
  result = *v3;
  if ( !*v3 )
  {
    result = sub_1A94A00(v13[0]);
    *v3 = result;
  }
  v5 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(a2 + 8);
    v7 = (v5 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( result == *v8 )
    {
LABEL_5:
      if ( v8 != (__int64 *)(v6 + 16 * v5) )
      {
        v10 = *(_QWORD *)(a2 + 32) + 16LL * *((unsigned int *)v8 + 2);
        if ( *(_QWORD *)(a2 + 40) != v10 )
          return *(_QWORD *)(v10 + 8);
      }
    }
    else
    {
      v11 = 1;
      while ( v9 != -8 )
      {
        v12 = v11 + 1;
        v7 = (v5 - 1) & (v11 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( result == *v8 )
          goto LABEL_5;
        v11 = v12;
      }
    }
  }
  return result;
}
