// Function: sub_3963C90
// Address: 0x3963c90
//
__int64 __fastcall sub_3963C90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // r15
  __int64 result; // rax
  int v9; // edx
  __int64 v10; // rsi
  int v11; // edi
  unsigned int v12; // eax
  __int64 v13; // rcx
  char v14; // r8
  __int64 v15; // rax
  unsigned int v16; // eax
  char v17; // cl
  __int64 v18; // rax
  int v19; // esi
  int v20; // edx
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v24[7]; // [rsp+8h] [rbp-38h] BYREF

  v23 = a3;
  if ( (unsigned __int8)sub_39538E0(a1 + 112, &v23, v24) )
    v5 = v24[0];
  else
    v5 = *(_QWORD *)(a1 + 120) + 16LL * *(unsigned int *)(a1 + 136);
  v6 = *(_DWORD *)(a1 + 80);
  v7 = *(_QWORD *)(v5 + 8);
  v23 = a2;
  result = 0;
  if ( v6 )
  {
    v9 = v6 - 1;
    v10 = *(_QWORD *)(a1 + 64);
    v11 = 1;
    v12 = v9 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = *(_QWORD *)(v10 + 16LL * v12);
    if ( a2 == v13 )
    {
LABEL_5:
      v14 = sub_1BFD9C0(a1 + 56, &v23, v24);
      v15 = v24[0];
      if ( v14 )
      {
        v16 = *(_DWORD *)(v24[0] + 8LL);
        v17 = v16 & 0x3F;
        v18 = 8LL * (v16 >> 6);
        return (*(_QWORD *)(*(_QWORD *)(v7 + 24) + v18) >> v17) & 1LL;
      }
      v19 = *(_DWORD *)(a1 + 72);
      ++*(_QWORD *)(a1 + 56);
      v20 = v19 + 1;
      v21 = *(_DWORD *)(a1 + 80);
      if ( 4 * v20 >= 3 * v21 )
      {
        v21 *= 2;
      }
      else if ( v21 - *(_DWORD *)(a1 + 76) - v20 > v21 >> 3 )
      {
LABEL_12:
        *(_DWORD *)(a1 + 72) = v20;
        if ( *(_QWORD *)v15 != -8 )
          --*(_DWORD *)(a1 + 76);
        v22 = v23;
        v17 = 0;
        *(_DWORD *)(v15 + 8) = 0;
        *(_QWORD *)v15 = v22;
        v18 = 0;
        return (*(_QWORD *)(*(_QWORD *)(v7 + 24) + v18) >> v17) & 1LL;
      }
      sub_1BFE340(a1 + 56, v21);
      sub_1BFD9C0(a1 + 56, &v23, v24);
      v15 = v24[0];
      v20 = *(_DWORD *)(a1 + 72) + 1;
      goto LABEL_12;
    }
    while ( v13 != -8 )
    {
      v12 = v9 & (v11 + v12);
      v13 = *(_QWORD *)(v10 + 16LL * v12);
      if ( a2 == v13 )
        goto LABEL_5;
      ++v11;
    }
    return 0;
  }
  return result;
}
