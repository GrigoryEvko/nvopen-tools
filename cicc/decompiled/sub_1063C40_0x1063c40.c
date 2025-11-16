// Function: sub_1063C40
// Address: 0x1063c40
//
__int64 __fastcall sub_1063C40(__int64 a1, __int64 a2)
{
  unsigned int v3; // ebx
  __int64 v4; // r12
  int v5; // esi
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned int v9; // ebx
  unsigned int i; // r12d
  __int64 *v11; // r13
  __int64 result; // rax
  __int64 v13; // r8
  int v14; // edx
  __int64 *v15; // rdi
  __int64 v16; // r9
  int v17; // edi
  int v18; // r10d
  int v19; // eax
  bool v20; // al
  unsigned int v21; // r12d
  __int64 v22; // [rsp+8h] [rbp-68h]
  int v23; // [rsp+14h] [rbp-5Ch]
  __int64 v24; // [rsp+18h] [rbp-58h]
  __int64 *v25; // [rsp+20h] [rbp-50h]
  __int64 v26; // [rsp+28h] [rbp-48h] BYREF
  __int64 *v27; // [rsp+38h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a1 + 56);
  v26 = a2;
  if ( v3 )
  {
    v8 = *(_QWORD *)(a1 + 40);
    v9 = v3 - 1;
    v24 = sub_1061AC0();
    v22 = sub_1061AD0();
    v23 = 1;
    v25 = 0;
    for ( i = v9 & sub_1061E50(v26); ; i = v9 & v21 )
    {
      v11 = (__int64 *)(v8 + 8LL * i);
      if ( sub_1061B40(v26, *v11) )
        break;
      if ( sub_1061B40(*v11, v24) )
      {
        v3 = *(_DWORD *)(a1 + 56);
        v4 = a1 + 32;
        if ( v25 )
          v11 = v25;
        v19 = *(_DWORD *)(a1 + 48);
        ++*(_QWORD *)(a1 + 32);
        v6 = v19 + 1;
        v27 = v11;
        if ( 4 * v6 >= 3 * v3 )
          goto LABEL_3;
        if ( v3 - (v6 + *(_DWORD *)(a1 + 52)) > v3 >> 3 )
          goto LABEL_5;
        v5 = v3;
        goto LABEL_4;
      }
      v20 = sub_1061B40(*v11, v22);
      if ( !v25 )
      {
        if ( !v20 )
          v11 = 0;
        v25 = v11;
      }
      v21 = v23 + i;
      ++v23;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 32);
    v4 = a1 + 32;
    v27 = 0;
LABEL_3:
    v5 = 2 * v3;
LABEL_4:
    sub_1063990(v4, v5);
    sub_1062220(v4, &v26, &v27);
    v6 = *(_DWORD *)(a1 + 48) + 1;
LABEL_5:
    *(_DWORD *)(a1 + 48) = v6;
    v7 = sub_1061AC0();
    if ( !sub_1061B40(*v27, v7) )
      --*(_DWORD *)(a1 + 52);
    *v27 = v26;
  }
  result = *(unsigned int *)(a1 + 24);
  v13 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)result )
  {
    v14 = result - 1;
    result = ((_DWORD)result - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v15 = (__int64 *)(v13 + 8 * result);
    v16 = *v15;
    if ( v26 == *v15 )
    {
LABEL_12:
      *v15 = -8192;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v17 = 1;
      while ( v16 != -4096 )
      {
        v18 = v17 + 1;
        result = v14 & (unsigned int)(v17 + result);
        v15 = (__int64 *)(v13 + 8LL * (unsigned int)result);
        v16 = *v15;
        if ( v26 == *v15 )
          goto LABEL_12;
        v17 = v18;
      }
    }
  }
  return result;
}
