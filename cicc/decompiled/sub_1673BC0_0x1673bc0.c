// Function: sub_1673BC0
// Address: 0x1673bc0
//
bool __fastcall sub_1673BC0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r12d
  __int64 v4; // rdi
  int v5; // r12d
  __int64 v6; // r14
  int v7; // r12d
  __int64 *v8; // r13
  int v9; // eax
  __int64 v10; // rax
  bool result; // al
  unsigned int v12; // r12d
  __int64 *v13; // r14
  int v14; // eax
  int v15; // r12d
  int v16; // r12d
  unsigned int i; // r14d
  bool v18; // al
  bool v19; // al
  bool v20; // al
  unsigned int v21; // r14d
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  int v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  int v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  int v30; // [rsp+18h] [rbp-48h]
  __int64 *v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  unsigned int v33; // [rsp+28h] [rbp-38h]
  unsigned int v34; // [rsp+28h] [rbp-38h]
  __int64 v35; // [rsp+28h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 56);
  if ( v3 )
  {
    v12 = v3 - 1;
    v8 = 0;
    v32 = *(_QWORD *)(a1 + 40);
    v29 = sub_16704E0();
    v23 = sub_16704F0();
    v26 = 1;
    v34 = v12 & sub_16707B0(a2);
    while ( 1 )
    {
      v13 = (__int64 *)(v32 + 8LL * v34);
      result = sub_1670560(a2, *v13);
      if ( result )
        return result;
      if ( sub_1670560(*v13, v29) )
        break;
      v18 = sub_1670560(*v13, v23);
      if ( !v8 && v18 )
        v8 = (__int64 *)(v32 + 8LL * v34);
      v34 = v12 & (v26 + v34);
      ++v26;
    }
    v14 = *(_DWORD *)(a1 + 48);
    v3 = *(_DWORD *)(a1 + 56);
    v4 = a1 + 32;
    if ( !v8 )
      v8 = (__int64 *)(v32 + 8LL * v34);
    ++*(_QWORD *)(a1 + 32);
    v9 = v14 + 1;
    if ( 4 * v9 < 3 * v3 )
    {
      if ( v3 - (v9 + *(_DWORD *)(a1 + 52)) > v3 >> 3 )
        goto LABEL_7;
      sub_16735E0(v4, v3);
      v15 = *(_DWORD *)(a1 + 56);
      if ( v15 )
      {
        v16 = v15 - 1;
        v35 = *(_QWORD *)(a1 + 40);
        v27 = sub_16704E0();
        v24 = sub_16704F0();
        v30 = 1;
        v31 = 0;
        for ( i = v16 & sub_16707B0(a2); ; i = v16 & v21 )
        {
          v8 = (__int64 *)(v35 + 8LL * i);
          if ( sub_1670560(a2, *v8) )
            break;
          if ( sub_1670560(*v8, v27) )
          {
LABEL_21:
            v9 = *(_DWORD *)(a1 + 48) + 1;
            if ( v31 )
              v8 = v31;
            goto LABEL_7;
          }
          v20 = sub_1670560(*v8, v24);
          if ( v31 || !v20 )
            v8 = v31;
          v31 = v8;
          v21 = v30 + i;
          ++v30;
        }
        goto LABEL_6;
      }
LABEL_37:
      ++*(_DWORD *)(a1 + 48);
      sub_16704E0();
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 32);
    v4 = a1 + 32;
  }
  sub_16735E0(v4, 2 * v3);
  v5 = *(_DWORD *)(a1 + 56);
  if ( !v5 )
    goto LABEL_37;
  v6 = *(_QWORD *)(a1 + 40);
  v7 = v5 - 1;
  v25 = sub_16704E0();
  v22 = sub_16704F0();
  v28 = 1;
  v31 = 0;
  v33 = v7 & sub_16707B0(a2);
  while ( 1 )
  {
    v8 = (__int64 *)(v6 + 8LL * v33);
    if ( sub_1670560(a2, *v8) )
      break;
    if ( sub_1670560(*v8, v25) )
      goto LABEL_21;
    v19 = sub_1670560(*v8, v22);
    if ( v31 || !v19 )
      v8 = v31;
    v31 = v8;
    v33 = v7 & (v28 + v33);
    ++v28;
  }
LABEL_6:
  v9 = *(_DWORD *)(a1 + 48) + 1;
LABEL_7:
  *(_DWORD *)(a1 + 48) = v9;
  v10 = sub_16704E0();
  result = sub_1670560(*v8, v10);
  if ( !result )
    --*(_DWORD *)(a1 + 52);
  *v8 = a2;
  return result;
}
