// Function: sub_FEF2D0
// Address: 0xfef2d0
//
__int64 __fastcall sub_FEF2D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // eax
  __int64 v7; // rcx
  int v8; // r8d
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 result; // rax
  int v13; // eax
  int v14; // r10d

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = -1;
  v6 = *(_DWORD *)(a3 + 24);
  *(_QWORD *)a1 = a2;
  v7 = *(_QWORD *)(a3 + 8);
  if ( !v6 )
    goto LABEL_7;
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a2 != *v10 )
  {
    v13 = 1;
    while ( v11 != -4096 )
    {
      v14 = v13 + 1;
      v9 = v8 & (v13 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v13 = v14;
    }
    goto LABEL_7;
  }
LABEL_3:
  result = v10[1];
  *(_QWORD *)(a1 + 8) = result;
  if ( !result )
  {
LABEL_7:
    result = sub_FEEEB0(a4, a2);
    *(_DWORD *)(a1 + 16) = result;
  }
  return result;
}
