// Function: sub_35A2540
// Address: 0x35a2540
//
__int64 __fastcall sub_35A2540(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 *v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // r11d
  __int64 *v11; // rdi
  unsigned int i; // eax
  __int64 *v13; // r8
  __int64 v14; // r10
  unsigned int v15; // eax
  __int64 v16; // rdx
  int v18; // eax
  int v19; // edx
  __int64 v20; // rax
  unsigned __int64 v21; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v22; // [rsp+8h] [rbp-38h] BYREF
  __int64 v23; // [rsp+10h] [rbp-30h] BYREF
  __int64 v24; // [rsp+18h] [rbp-28h]

  v21 = sub_2EBEE90(*(_QWORD *)(a1 + 24), a2);
  v5 = (unsigned int)sub_2E8E710(v21, a2, 0, 0, 0);
  v6 = sub_359C4A0(a1 + 256, (__int64 *)&v21);
  v23 = a3;
  v7 = *(_DWORD *)(a1 + 312);
  v8 = *v6;
  v24 = *v6;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 288);
    v22 = 0;
    goto LABEL_24;
  }
  v10 = 1;
  v11 = 0;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
              | ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))); ; i = (v7 - 1) & v15 )
  {
    v9 = *(_QWORD *)(a1 + 296);
    v13 = (__int64 *)(v9 + 24LL * i);
    v14 = *v13;
    if ( a3 == *v13 && v8 == v13[1] )
    {
      v16 = v13[2];
      return *(unsigned int *)(*(_QWORD *)(v16 + 32) + 40 * v5 + 8);
    }
    if ( v14 == -4096 )
      break;
    if ( v14 == -8192 && v13[1] == -8192 && !v11 )
      v11 = (__int64 *)(v9 + 24LL * i);
LABEL_9:
    v15 = v10 + i;
    ++v10;
  }
  if ( v13[1] != -4096 )
    goto LABEL_9;
  v18 = *(_DWORD *)(a1 + 304);
  if ( !v11 )
    v11 = v13;
  ++*(_QWORD *)(a1 + 288);
  v19 = v18 + 1;
  v22 = v11;
  if ( 4 * (v18 + 1) < 3 * v7 )
  {
    if ( v7 - *(_DWORD *)(a1 + 308) - v19 > v7 >> 3 )
      goto LABEL_18;
    goto LABEL_25;
  }
LABEL_24:
  v7 *= 2;
LABEL_25:
  sub_35A1120(a1 + 288, v7);
  sub_359BDE0(a1 + 288, &v23, &v22);
  a3 = v23;
  v11 = v22;
  v19 = *(_DWORD *)(a1 + 304) + 1;
LABEL_18:
  *(_DWORD *)(a1 + 304) = v19;
  if ( *v11 != -4096 || v11[1] != -4096 )
    --*(_DWORD *)(a1 + 308);
  *v11 = a3;
  v20 = v24;
  v16 = 0;
  v11[2] = 0;
  v11[1] = v20;
  return *(unsigned int *)(*(_QWORD *)(v16 + 32) + 40 * v5 + 8);
}
