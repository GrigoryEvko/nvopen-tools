// Function: sub_1DFBF70
// Address: 0x1dfbf70
//
__int64 *__fastcall sub_1DFBF70(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 v4; // rdi
  int v5; // r9d
  unsigned int v6; // edx
  __int64 v7; // rcx
  __int64 *v8; // r12
  __int64 *v9; // rbx
  __int64 v10; // r13
  __int64 *v11; // r14
  unsigned int v12; // r15d
  int v13; // esi
  __int64 *v14; // rcx
  int v15; // eax
  __int64 v16; // rax
  int v17; // eax
  unsigned int v18; // r8d
  __int64 *v19; // r11
  int v20; // r14d
  unsigned int i; // r13d
  __int64 v22; // rdi
  __int64 *v23; // r15
  __int64 v24; // r10
  char v25; // al
  __int64 v26; // rax
  int v28; // eax
  char v29; // al
  unsigned int v30; // r13d
  __int64 v31; // [rsp+0h] [rbp-80h]
  __int64 *v32; // [rsp+8h] [rbp-78h]
  __int64 *v33; // [rsp+8h] [rbp-78h]
  unsigned int v34; // [rsp+14h] [rbp-6Ch]
  unsigned int v35; // [rsp+14h] [rbp-6Ch]
  __int64 v36; // [rsp+18h] [rbp-68h]
  __int64 v37; // [rsp+30h] [rbp-50h]
  __int64 *v38; // [rsp+38h] [rbp-48h]
  __int64 *v39[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = a1;
  v3 = *(unsigned int *)(a1 + 328);
  v4 = *(_QWORD *)(a1 + 312);
  if ( !(_DWORD)v3 )
  {
LABEL_27:
    v38 = (__int64 *)(v4 + 16 * v3);
    goto LABEL_3;
  }
  v5 = 1;
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v38 = (__int64 *)(v4 + 16LL * v6);
  v7 = *v38;
  if ( a2 != *v38 )
  {
    while ( v7 != -8 )
    {
      v6 = (v3 - 1) & (v5 + v6);
      v38 = (__int64 *)(v4 + 16LL * v6);
      v7 = *v38;
      if ( a2 == *v38 )
        goto LABEL_3;
      ++v5;
    }
    goto LABEL_27;
  }
LABEL_3:
  v8 = (__int64 *)v38[1];
  if ( !v8 )
    goto LABEL_23;
  *(_QWORD *)(*v8 + 32) = v8[1];
  v9 = (__int64 *)v8[2];
  if ( !v9 )
    goto LABEL_22;
  v31 = v2;
  do
  {
    v10 = *v8;
    v11 = v9 + 2;
    if ( !v9[1] )
    {
      if ( (unsigned __int8)sub_1DF9540(*v8, v9 + 2, v39) )
      {
        *v39[0] = -1;
        --*(_DWORD *)(v10 + 16);
        ++*(_DWORD *)(v10 + 20);
      }
      goto LABEL_20;
    }
    v12 = *(_DWORD *)(v10 + 24);
    if ( !v12 )
    {
      ++*(_QWORD *)v10;
LABEL_9:
      v13 = 2 * v12;
LABEL_10:
      sub_1DFBB90(v10, v13);
      sub_1DF9540(v10, v11, v39);
      v14 = v39[0];
      v15 = *(_DWORD *)(v10 + 16) + 1;
      goto LABEL_11;
    }
    v37 = *(_QWORD *)(v10 + 8);
    v17 = sub_1E1C690(v9 + 2);
    v18 = v12 - 1;
    v19 = 0;
    v36 = v10;
    v20 = 1;
    for ( i = (v12 - 1) & v17; ; i = v35 & v30 )
    {
      v22 = v9[2];
      v23 = (__int64 *)(v37 + 16LL * i);
      v24 = *v23;
      if ( (unsigned __int64)(v22 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(v24 - 1) > 0xFFFFFFFFFFFFFFFDLL )
      {
        if ( v22 == v24 )
        {
LABEL_18:
          v14 = (__int64 *)(v37 + 16LL * i);
          goto LABEL_19;
        }
      }
      else
      {
        v32 = v19;
        v34 = v18;
        v25 = sub_1E15D60(v22, *v23, 3);
        v18 = v34;
        v19 = v32;
        if ( v25 )
          goto LABEL_18;
        v24 = *v23;
      }
      v33 = v19;
      v35 = v18;
      if ( sub_1DF7390(v24, 0) )
        break;
      v29 = sub_1DF7390(*v23, -1);
      v19 = v33;
      v18 = v35;
      if ( !v33 && v29 )
        v19 = (__int64 *)(v37 + 16LL * i);
      v30 = v20 + i;
      ++v20;
    }
    v10 = v36;
    v14 = v23;
    v11 = v9 + 2;
    if ( v33 )
      v14 = v33;
    v28 = *(_DWORD *)(v36 + 16);
    v12 = *(_DWORD *)(v36 + 24);
    ++*(_QWORD *)v36;
    v15 = v28 + 1;
    if ( 4 * v15 >= 3 * v12 )
      goto LABEL_9;
    if ( v12 - (v15 + *(_DWORD *)(v36 + 20)) <= v12 >> 3 )
    {
      v13 = v12;
      goto LABEL_10;
    }
LABEL_11:
    *(_DWORD *)(v10 + 16) = v15;
    if ( *v14 )
      --*(_DWORD *)(v10 + 20);
    v16 = v9[2];
    v14[1] = 0;
    *v14 = v16;
LABEL_19:
    v14[1] = v9[1];
LABEL_20:
    v8[2] = *v9;
    v26 = *v8;
    *v9 = *(_QWORD *)(*v8 + 40);
    *(_QWORD *)(v26 + 40) = v9;
    v9 = (__int64 *)v8[2];
  }
  while ( v9 );
  v2 = v31;
LABEL_22:
  j_j___libc_free_0(v8, 24);
LABEL_23:
  *v38 = -16;
  --*(_DWORD *)(v2 + 320);
  ++*(_DWORD *)(v2 + 324);
  return v38;
}
