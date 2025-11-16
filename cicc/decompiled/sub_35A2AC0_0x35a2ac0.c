// Function: sub_35A2AC0
// Address: 0x35a2ac0
//
__int64 __fastcall sub_35A2AC0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned int v6; // esi
  __int64 v7; // r8
  __int64 v8; // rdi
  int v9; // r14d
  __int64 *v10; // r10
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r11
  int v14; // r12d
  __int64 v15; // rdx
  int v16; // r14d
  __int64 v17; // rax
  unsigned int v18; // r15d
  unsigned __int64 v19; // rax
  int v21; // eax
  int v22; // edx
  __int64 v23; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = a1 + 224;
  v6 = *(_DWORD *)(a1 + 248);
  v23 = a3;
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 224);
    v24[0] = 0;
    goto LABEL_25;
  }
  v7 = *(_QWORD *)(a1 + 232);
  v8 = a3;
  v9 = 1;
  v10 = 0;
  v11 = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v12 = (__int64 *)(v7 + 16LL * v11);
  v13 = *v12;
  if ( v8 != *v12 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v10 )
        v10 = v12;
      v11 = (v6 - 1) & (v9 + v11);
      v12 = (__int64 *)(v7 + 16LL * v11);
      v13 = *v12;
      if ( v8 == *v12 )
        goto LABEL_3;
      ++v9;
    }
    if ( !v10 )
      v10 = v12;
    v21 = *(_DWORD *)(a1 + 240);
    ++*(_QWORD *)(a1 + 224);
    v22 = v21 + 1;
    v24[0] = v10;
    if ( 4 * (v21 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 244) - v22 > v6 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(a1 + 240) = v22;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a1 + 244);
        *v10 = v8;
        *((_DWORD *)v10 + 2) = 0;
        v15 = *(_QWORD *)(a2 + 32);
        return *(unsigned int *)(v15 + 8);
      }
LABEL_26:
      sub_2E261E0(v4, v6);
      sub_35472E0(v4, &v23, v24);
      v8 = v23;
      v10 = (__int64 *)v24[0];
      v22 = *(_DWORD *)(a1 + 240) + 1;
      goto LABEL_20;
    }
LABEL_25:
    v6 *= 2;
    goto LABEL_26;
  }
LABEL_3:
  v14 = *((_DWORD *)v12 + 2);
  v15 = *(_QWORD *)(a2 + 32);
  if ( !v14 )
    return *(unsigned int *)(v15 + 8);
  v16 = 0;
  while ( 1 )
  {
    v17 = 40;
    if ( *(_QWORD *)(a2 + 24) != *(_QWORD *)(v15 + 104) )
      v17 = 120;
    ++v16;
    v18 = *(_DWORD *)(v15 + v17 + 8);
    v19 = sub_2EBEE10(*(_QWORD *)(a1 + 24), v18);
    a2 = v19;
    if ( v16 == v14 )
      break;
    v15 = *(_QWORD *)(v19 + 32);
  }
  return v18;
}
