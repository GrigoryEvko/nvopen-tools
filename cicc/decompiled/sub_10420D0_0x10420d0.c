// Function: sub_10420D0
// Address: 0x10420d0
//
__int64 __fastcall sub_10420D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rax
  int v5; // r14d
  _QWORD *v6; // r15
  __int64 v7; // r12
  __int64 v8; // rax
  int v9; // eax
  unsigned int v10; // esi
  __int64 v11; // rdi
  int v12; // r11d
  __int64 *v13; // r9
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // rax
  int v19; // eax
  int v20; // edx
  __int64 v21; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v22; // [rsp+8h] [rbp-38h] BYREF

  v2 = a2;
  v4 = sub_AA48A0(a2);
  v5 = *(_DWORD *)(a1 + 352);
  v6 = (_QWORD *)v4;
  *(_DWORD *)(a1 + 352) = v5 + 1;
  v7 = sub_BD2DA0(80);
  if ( v7 )
  {
    v8 = sub_BCB120(v6);
    sub_BD35F0(v7, v8, 28);
    v9 = *(_DWORD *)(v7 + 4);
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 64) = a2;
    *(_DWORD *)(v7 + 72) = v5;
    *(_DWORD *)(v7 + 4) = v9 & 0x38000000 | 0x40000000;
    *(_QWORD *)(v7 + 40) = 0;
    *(_QWORD *)(v7 + 24) = sub_103AB10;
    *(_QWORD *)(v7 + 48) = 0;
    *(_QWORD *)(v7 + 56) = 0;
    *(_DWORD *)(v7 + 76) = 0;
    sub_BD2A10(v7, 0, 1);
  }
  sub_1041C60(a1, v7, a2, 0);
  v10 = *(_DWORD *)(a1 + 56);
  v21 = v2;
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 32);
    v22 = 0;
LABEL_21:
    v10 *= 2;
    goto LABEL_22;
  }
  v11 = *(_QWORD *)(a1 + 40);
  v12 = 1;
  v13 = 0;
  v14 = (v10 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v15 = (__int64 *)(v11 + 16LL * v14);
  v16 = *v15;
  if ( v2 == *v15 )
  {
LABEL_5:
    v17 = v15 + 1;
    goto LABEL_6;
  }
  while ( v16 != -4096 )
  {
    if ( v16 == -8192 && !v13 )
      v13 = v15;
    v14 = (v10 - 1) & (v12 + v14);
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( v2 == *v15 )
      goto LABEL_5;
    ++v12;
  }
  if ( !v13 )
    v13 = v15;
  v19 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  v20 = v19 + 1;
  v22 = v13;
  if ( 4 * (v19 + 1) >= 3 * v10 )
    goto LABEL_21;
  if ( v10 - *(_DWORD *)(a1 + 52) - v20 <= v10 >> 3 )
  {
LABEL_22:
    sub_103FE70(a1 + 32, v10);
    sub_103F430(a1 + 32, &v21, &v22);
    v2 = v21;
    v13 = v22;
    v20 = *(_DWORD *)(a1 + 48) + 1;
  }
  *(_DWORD *)(a1 + 48) = v20;
  if ( *v13 != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v13 = v2;
  v17 = v13 + 1;
  v13[1] = 0;
LABEL_6:
  *v17 = v7;
  return v7;
}
