// Function: sub_AC4320
// Address: 0xac4320
//
__int64 __fastcall sub_AC4320(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 result; // rax
  int v5; // edx
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  int v11; // edx
  int v12; // r9d

  v3 = *(_QWORD *)(a1 - 32);
  result = *(_QWORD *)sub_BD5C60(v3, a2, a3);
  v5 = *(_DWORD *)(result + 2080);
  v6 = *(_QWORD *)(result + 2064);
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( v3 == *v9 )
    {
LABEL_3:
      *v9 = -8192;
      --*(_DWORD *)(result + 2072);
      ++*(_DWORD *)(result + 2076);
    }
    else
    {
      v11 = 1;
      while ( v10 != -4096 )
      {
        v12 = v11 + 1;
        v8 = v7 & (v11 + v8);
        v9 = (__int64 *)(v6 + 16LL * v8);
        v10 = *v9;
        if ( v3 == *v9 )
          goto LABEL_3;
        v11 = v12;
      }
    }
  }
  return result;
}
