// Function: sub_19E7150
// Address: 0x19e7150
//
__int64 __fastcall sub_19E7150(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  int v4; // r8d
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 *v9; // r12
  __int64 v10; // rsi
  __int64 v11; // r15
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  _QWORD v16[11]; // [rsp-58h] [rbp-58h] BYREF

  v3 = *(unsigned int *)(a1 + 1984);
  if ( !(_DWORD)v3 )
    return 0;
  v4 = 1;
  v7 = *(_QWORD *)(a1 + 1968);
  v8 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    while ( v10 != -8 )
    {
      v8 = (v3 - 1) & (v4 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      ++v4;
    }
    return 0;
  }
LABEL_3:
  if ( v9 == (__int64 *)(v7 + 16 * v3) )
    return 0;
  v11 = v9[1];
  if ( v11 == a3 )
    return 0;
  if ( *(_BYTE *)(a2 + 16) == 23 )
  {
    sub_19E4680(v11 + 128, a2);
    v12 = sub_1412190(a3 + 128, a2);
    v13 = *(_QWORD *)(a3 + 144);
    v14 = v13 == *(_QWORD *)(a3 + 136) ? *(unsigned int *)(a3 + 156) : *(unsigned int *)(a3 + 152);
    v16[0] = v12;
    v16[1] = v13 + 8 * v14;
    sub_19E4730((__int64)v16);
    if ( a2 == *(_QWORD *)(v11 + 40) )
    {
      if ( *(_DWORD *)(v11 + 184) || *(_DWORD *)(v11 + 156) != *(_DWORD *)(v11 + 160) )
      {
        *(_QWORD *)(v11 + 40) = sub_19E6D50(a1, v11);
        sub_19E4760(a1, v11);
      }
      else
      {
        *(_QWORD *)(v11 + 40) = 0;
      }
    }
  }
  v9[1] = a3;
  return 1;
}
