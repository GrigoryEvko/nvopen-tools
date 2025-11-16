// Function: sub_13FA5B0
// Address: 0x13fa5b0
//
_QWORD *__fastcall sub_13FA5B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *result; // rax
  __int64 v6; // rdi
  __int64 v7; // r13
  unsigned int v8; // r12d
  _QWORD *v9; // r14
  _QWORD *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rdx
  unsigned int v15; // edx
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  _QWORD *v19; // rdx
  _QWORD *v20; // [rsp+0h] [rbp-60h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  _QWORD *v22; // [rsp+10h] [rbp-50h]
  __int64 v23; // [rsp+18h] [rbp-48h]
  int v24; // [rsp+24h] [rbp-3Ch]
  __int64 v25; // [rsp+28h] [rbp-38h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v3 = a1 + 56;
  result = *(_QWORD **)(v3 - 16);
  v23 = v3;
  v20 = result;
  v22 = *(_QWORD **)(v3 - 24);
  if ( v22 != result )
  {
    while ( 1 )
    {
      v21 = *v22;
      v6 = sub_157EBA0(*v22);
      if ( v6 )
      {
        v24 = sub_15F4D60(v6);
        v7 = sub_157EBA0(v21);
        if ( v24 )
          break;
      }
LABEL_25:
      result = ++v22;
      if ( v20 == v22 )
        return result;
    }
    v8 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = sub_15F4DF0(v7, v8);
        v14 = *(_QWORD **)(a1 + 72);
        v11 = v13;
        v10 = *(_QWORD **)(a1 + 64);
        if ( v14 == v10 )
          break;
        v25 = v11;
        v9 = &v14[*(unsigned int *)(a1 + 80)];
        v10 = (_QWORD *)sub_16CC9F0(v23, v11);
        v11 = v25;
        if ( v25 == *v10 )
        {
          v17 = *(_QWORD *)(a1 + 72);
          if ( v17 == *(_QWORD *)(a1 + 64) )
            v18 = *(unsigned int *)(a1 + 84);
          else
            v18 = *(unsigned int *)(a1 + 80);
          v19 = (_QWORD *)(v17 + 8 * v18);
          goto LABEL_18;
        }
        v12 = *(_QWORD *)(a1 + 72);
        if ( v12 == *(_QWORD *)(a1 + 64) )
        {
          v10 = (_QWORD *)(v12 + 8LL * *(unsigned int *)(a1 + 84));
          v19 = v10;
          goto LABEL_18;
        }
        v10 = (_QWORD *)(v12 + 8LL * *(unsigned int *)(a1 + 80));
LABEL_8:
        if ( v9 == v10 )
          goto LABEL_20;
LABEL_9:
        if ( v24 == ++v8 )
          goto LABEL_25;
      }
      v9 = &v10[*(unsigned int *)(a1 + 84)];
      if ( v10 == v9 )
      {
        v19 = *(_QWORD **)(a1 + 64);
      }
      else
      {
        do
        {
          if ( v11 == *v10 )
            break;
          ++v10;
        }
        while ( v9 != v10 );
        v19 = v9;
      }
LABEL_18:
      while ( v19 != v10 )
      {
        if ( *v10 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_8;
        ++v10;
      }
      if ( v9 != v10 )
        goto LABEL_9;
LABEL_20:
      v15 = *(_DWORD *)(a2 + 8);
      if ( v15 >= *(_DWORD *)(a2 + 12) )
      {
        v26 = v11;
        sub_16CD150(a2, a2 + 16, 0, 16);
        v15 = *(_DWORD *)(a2 + 8);
        v11 = v26;
      }
      v16 = (_QWORD *)(*(_QWORD *)a2 + 16LL * v15);
      if ( v16 )
      {
        v16[1] = v11;
        *v16 = v21;
        v15 = *(_DWORD *)(a2 + 8);
      }
      ++v8;
      *(_DWORD *)(a2 + 8) = v15 + 1;
      if ( v24 == v8 )
        goto LABEL_25;
    }
  }
  return result;
}
