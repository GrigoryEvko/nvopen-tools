// Function: sub_27801B0
// Address: 0x27801b0
//
_QWORD *__fastcall sub_27801B0(__int64 a1, __int64 a2, unsigned __int8 **a3, _QWORD *a4)
{
  __int64 *v4; // r8
  _QWORD *v5; // r10
  unsigned int v6; // r12d
  __int64 v7; // r13
  __int64 *v8; // r14
  int v9; // eax
  __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *result; // rax
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // rcx
  unsigned int v16; // r12d
  int v17; // eax
  __int64 *v18; // r13
  char v19; // al
  __int64 v20; // rdx
  int v21; // eax
  int v22; // r13d
  int v23; // [rsp+Ch] [rbp-64h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  _QWORD *v26; // [rsp+18h] [rbp-58h]
  _QWORD *v27; // [rsp+20h] [rbp-50h]
  unsigned __int8 **v29; // [rsp+20h] [rbp-50h]
  _QWORD *v30; // [rsp+20h] [rbp-50h]
  unsigned __int8 **v31; // [rsp+28h] [rbp-48h]
  unsigned int i; // [rsp+28h] [rbp-48h]
  __int64 *v33; // [rsp+28h] [rbp-48h]
  __int64 *v34[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = (__int64 *)a3;
  v5 = a4;
  v6 = *(_DWORD *)(a1 + 128);
  if ( v6 )
  {
    v16 = v6 - 1;
    v8 = 0;
    v24 = *(_QWORD *)(a1 + 112);
    v17 = sub_277F590(*a3);
    v23 = 1;
    v5 = a4;
    v4 = (__int64 *)a3;
    for ( i = v16 & v17; ; i = v22 )
    {
      v26 = v5;
      v29 = (unsigned __int8 **)v4;
      v18 = (__int64 *)(v24 + 16LL * i);
      v19 = sub_277AC50(*v4, *v18);
      v4 = (__int64 *)v29;
      v5 = v26;
      if ( v19 )
      {
        result = *(_QWORD **)a1;
        v13 = v18[1];
        v11 = v18 + 1;
        v14 = *(_QWORD *)(a2 + 16);
        if ( !*(_QWORD *)a1 )
          goto LABEL_7;
        goto LABEL_14;
      }
      if ( *v18 == -4096 )
        break;
      if ( !v8 && *v18 == -8192 )
        v8 = (__int64 *)(v24 + 16LL * i);
      v22 = v16 & (v23 + i);
      ++v23;
    }
    v21 = *(_DWORD *)(a1 + 120);
    v6 = *(_DWORD *)(a1 + 128);
    if ( !v8 )
      v8 = (__int64 *)(v24 + 16LL * i);
    ++*(_QWORD *)(a1 + 104);
    v7 = a1 + 104;
    v9 = v21 + 1;
    v34[0] = v8;
    if ( 4 * v9 < 3 * v6 )
    {
      if ( v6 - (v9 + *(_DWORD *)(a1 + 124)) <= v6 >> 3 )
      {
        sub_277FB40(a1 + 104, v6);
        sub_277FA60(a1 + 104, v29, v34);
        v8 = v34[0];
        v5 = v26;
        v4 = (__int64 *)v29;
        v9 = *(_DWORD *)(a1 + 120) + 1;
      }
      goto LABEL_4;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 104);
    v7 = a1 + 104;
    v34[0] = 0;
  }
  v27 = v5;
  v31 = (unsigned __int8 **)v4;
  sub_277FB40(v7, 2 * v6);
  sub_277FA60(v7, v31, v34);
  v8 = v34[0];
  v4 = (__int64 *)v31;
  v5 = v27;
  v9 = *(_DWORD *)(a1 + 120) + 1;
LABEL_4:
  *(_DWORD *)(a1 + 120) = v9;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 124);
  v10 = *v4;
  v8[1] = 0;
  v11 = v8 + 1;
  *v8 = v10;
  result = *(_QWORD **)a1;
  v13 = 0;
  v14 = *(_QWORD *)(a2 + 16);
  if ( *(_QWORD *)a1 )
  {
LABEL_14:
    *(_QWORD *)a1 = *result;
  }
  else
  {
LABEL_7:
    v15 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 88) += 32LL;
    result = (_QWORD *)((v15 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( *(_QWORD *)(a1 + 16) >= (unsigned __int64)(result + 4) && v15 )
    {
      *(_QWORD *)(a1 + 8) = result + 4;
      if ( !result )
      {
        MEMORY[0] = v14;
        BUG();
      }
    }
    else
    {
      v30 = v5;
      v33 = v4;
      result = (_QWORD *)sub_9D1E70(a1 + 8, 32, 32, 3);
      v5 = v30;
      v4 = v33;
    }
  }
  result[2] = *v4;
  v20 = *v5;
  *result = v14;
  result[3] = v20;
  result[1] = v13;
  *v11 = result;
  *(_QWORD *)(a2 + 16) = result;
  return result;
}
