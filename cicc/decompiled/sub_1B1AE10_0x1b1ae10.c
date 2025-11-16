// Function: sub_1B1AE10
// Address: 0x1b1ae10
//
__int64 __fastcall sub_1B1AE10(__int64 a1, __int64 *a2, __int64 a3)
{
  int v4; // r8d
  int v5; // r9d
  int v6; // eax
  __int64 v7; // rax
  __int64 **v8; // r13
  _QWORD *v9; // rbx
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v13; // r12
  _QWORD *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rsi
  _QWORD *v18; // rdx
  __int64 v19; // rax
  const void *v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+10h] [rbp-50h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 **v23; // [rsp+20h] [rbp-40h]
  __int64 v24; // [rsp+28h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  v20 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  v22 = a3 + 56;
  if ( sub_1377F70(a3 + 56, *a2) )
  {
    v19 = *(unsigned int *)(a1 + 8);
    if ( (unsigned int)v19 >= *(_DWORD *)(a1 + 12) )
    {
      sub_16CD150(a1, v20, 0, 8, v4, v5);
      v19 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v19) = a2;
    v6 = *(_DWORD *)(a1 + 8) + 1;
    *(_DWORD *)(a1 + 8) = v6;
  }
  else
  {
    v6 = *(_DWORD *)(a1 + 8);
  }
  v21 = 0;
  if ( v6 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v21);
      v8 = *(__int64 ***)(v7 + 24);
      v23 = *(__int64 ***)(v7 + 32);
      if ( v23 != v8 )
        break;
LABEL_28:
      if ( ++v21 >= (unsigned __int64)*(unsigned int *)(a1 + 8) )
        return a1;
    }
    while ( 1 )
    {
      v13 = *v8;
      v14 = *(_QWORD **)(a3 + 72);
      v10 = *(_QWORD **)(a3 + 64);
      if ( v14 == v10 )
      {
        v9 = &v10[*(unsigned int *)(a3 + 84)];
        if ( v10 == v9 )
        {
          v18 = *(_QWORD **)(a3 + 64);
        }
        else
        {
          do
          {
            if ( **v8 == *v10 )
              break;
            ++v10;
          }
          while ( v9 != v10 );
          v18 = v9;
        }
        goto LABEL_22;
      }
      v24 = **v8;
      v9 = &v14[*(unsigned int *)(a3 + 80)];
      v10 = sub_16CC9F0(v22, v24);
      if ( v24 == *v10 )
        break;
      v11 = *(_QWORD *)(a3 + 72);
      if ( v11 == *(_QWORD *)(a3 + 64) )
      {
        v10 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(a3 + 84));
        v18 = v10;
LABEL_22:
        while ( v18 != v10 && *v10 >= 0xFFFFFFFFFFFFFFFELL )
          ++v10;
        goto LABEL_9;
      }
      v10 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(a3 + 80));
LABEL_9:
      if ( v9 != v10 )
      {
        v12 = *(unsigned int *)(a1 + 8);
        if ( (unsigned int)v12 >= *(_DWORD *)(a1 + 12) )
        {
          sub_16CD150(a1, v20, 0, 8, v4, v5);
          v12 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v12) = v13;
        ++*(_DWORD *)(a1 + 8);
      }
      if ( v23 == ++v8 )
        goto LABEL_28;
    }
    v15 = *(_QWORD *)(a3 + 72);
    if ( v15 == *(_QWORD *)(a3 + 64) )
      v16 = *(unsigned int *)(a3 + 84);
    else
      v16 = *(unsigned int *)(a3 + 80);
    v18 = (_QWORD *)(v15 + 8 * v16);
    goto LABEL_22;
  }
  return a1;
}
