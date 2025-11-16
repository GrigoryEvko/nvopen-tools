// Function: sub_B2FA10
// Address: 0xb2fa10
//
__int64 __fastcall sub_B2FA10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v5; // edx
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  int v11; // edx
  int v12; // r9d

  result = *(_QWORD *)sub_BD5C60(a1, a2, a3);
  v5 = *(_DWORD *)(result + 3376);
  v6 = *(_QWORD *)(result + 3360);
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( a1 == *v9 )
    {
LABEL_3:
      *v9 = -8192;
      --*(_DWORD *)(result + 3368);
      ++*(_DWORD *)(result + 3372);
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
        if ( a1 == *v9 )
          goto LABEL_3;
        v11 = v12;
      }
    }
  }
  *(_BYTE *)(a1 + 34) &= ~1u;
  return result;
}
