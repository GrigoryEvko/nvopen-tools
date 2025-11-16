// Function: sub_20430D0
// Address: 0x20430d0
//
__int64 __fastcall sub_20430D0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  int v5; // r12d
  __int64 *v6; // r15
  __int64 *v7; // r13
  __int64 result; // rax
  __int64 *v9; // rdx
  int v10; // ecx
  __int64 v11; // rdi
  __int64 *j; // rsi
  int v13; // edi
  int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r8
  __int64 *k; // rsi
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int8 v21; // si
  __int64 v22; // rdi
  __int64 v23; // r15
  __int64 (__fastcall *v24)(__int64, unsigned __int8); // rax
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // r15
  unsigned __int8 v29; // si
  __int64 v30; // rdi
  __int64 v31; // r8
  __int64 (__fastcall *v32)(__int64, unsigned __int8); // rax
  bool v33; // cc
  __int64 v34; // rax
  _QWORD *v35; // rax
  _QWORD *i; // rsi
  unsigned __int64 v37; // rdx
  __int16 v38; // cx
  unsigned int v39; // edx
  __int64 v40; // rax
  __int64 v41; // [rsp+0h] [rbp-40h]
  __int64 v42; // [rsp+8h] [rbp-38h]
  unsigned int v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]

  if ( a2 )
  {
    v3 = *(_QWORD *)a2;
    if ( *(__int16 *)(*(_QWORD *)a2 + 24LL) < 0 )
    {
      v18 = 0;
      v19 = *(unsigned int *)(v3 + 60);
      v20 = 16 * v19;
      if ( (_DWORD)v19 )
      {
        do
        {
          v21 = *(_BYTE *)(*(_QWORD *)(v3 + 40) + v18);
          if ( v21 )
          {
            v22 = *(_QWORD *)(a1 + 136);
            v23 = *(_QWORD *)(v22 + 8LL * v21 + 120);
            if ( v23 )
            {
              v24 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v22 + 288LL);
              if ( v24 == sub_1D45FB0 || (v44 = v20, v40 = ((__int64 (*)(void))v24)(), v20 = v44, (v23 = v40) != 0) )
              {
                v42 = v20;
                v25 = sub_2041EC0(a1, a2, *(unsigned __int16 *)(*(_QWORD *)v23 + 24LL));
                v20 = v42;
                *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * *(unsigned __int16 *)(*(_QWORD *)v23 + 24LL)) += v25;
              }
            }
          }
          v18 += 16;
        }
        while ( v20 != v18 );
      }
      v26 = *(unsigned int *)(v3 + 56);
      if ( (_DWORD)v26 )
      {
        v27 = 0;
        v28 = 40 * v26;
        do
        {
          v29 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v27 + *(_QWORD *)(v3 + 32)) + 40LL)
                         + 16LL * *(unsigned int *)(v27 + *(_QWORD *)(v3 + 32) + 8));
          if ( v29 )
          {
            v30 = *(_QWORD *)(a1 + 136);
            v31 = *(_QWORD *)(v30 + 8LL * v29 + 120);
            if ( v31 )
            {
              v32 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v30 + 288LL);
              if ( v32 == sub_1D45FB0 || (v31 = ((__int64 (*)(void))v32)()) != 0 )
              {
                v41 = v31;
                v43 = *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * *(unsigned __int16 *)(*(_QWORD *)v31 + 24LL));
                v33 = v43 <= (unsigned int)sub_2041DA0(a1, a2, *(unsigned __int16 *)(*(_QWORD *)v31 + 24LL));
                v34 = *(_QWORD *)v41;
                if ( v33 )
                  *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * *(unsigned __int16 *)(v34 + 24)) = 0;
                else
                  *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * *(unsigned __int16 *)(*(_QWORD *)v41 + 24LL)) -= sub_2041DA0(a1, a2, *(unsigned __int16 *)(v34 + 24));
              }
            }
          }
          v27 += 40;
        }
        while ( v28 != v27 );
      }
      v35 = *(_QWORD **)(a2 + 32);
      for ( i = &v35[2 * *(unsigned int *)(a2 + 40)]; i != v35; v35 += 2 )
      {
        if ( (*v35 & 6) == 0 )
        {
          v37 = *v35 & 0xFFFFFFFFFFFFFFF8LL;
          v38 = *(_WORD *)(v37 + 224);
          if ( v38 )
            *(_WORD *)(v37 + 224) = v38 - 1;
        }
      }
    }
    v5 = 0;
    sub_2042F40(a1, (__int64 *)a2);
    v6 = *(__int64 **)(a2 + 112);
    v7 = &v6[2 * *(unsigned int *)(a2 + 120)];
    if ( v6 == v7 )
      goto LABEL_33;
    do
    {
      sub_2042A20(a1, *v6 & 0xFFFFFFFFFFFFFFF8LL);
      v5 += ((*v6 >> 1) & 3) == 0;
      v6 += 2;
    }
    while ( v7 != v6 );
    if ( !v5 )
    {
LABEL_33:
      v39 = *(_DWORD *)(a2 + 200);
      result = 0;
      if ( *(_DWORD *)(a1 + 192) >= v39 )
        result = *(_DWORD *)(a1 + 192) - v39;
      *(_DWORD *)(a1 + 192) = result;
    }
    else
    {
      result = *(unsigned __int16 *)(a2 + 224);
      *(_DWORD *)(a1 + 192) += result;
    }
    v9 = *(__int64 **)(a2 + 112);
    v10 = 0;
    v11 = *(unsigned int *)(a2 + 120);
    for ( j = &v9[2 * v11]; j != v9; v9 += 2 )
    {
      result = (*v9 >> 1) & 3;
      v10 -= (result == 0) - 1;
    }
    v13 = *(_DWORD *)(a1 + 196) + v11 - v10;
    v14 = 0;
    *(_DWORD *)(a1 + 196) = v13;
    v15 = *(__int64 **)(a2 + 32);
    v16 = *(unsigned int *)(a2 + 40);
    for ( k = &v15[2 * v16]; k != v15; v15 += 2 )
    {
      result = (*v15 >> 1) & 3;
      v14 -= (result == 0) - 1;
    }
    *(_DWORD *)(a1 + 196) = v13 - v16 + v14;
  }
  else
  {
    *(_DWORD *)(*(_QWORD *)(a1 + 160) + 8LL) = 0;
    result = *(_QWORD *)(a1 + 168);
    if ( *(_QWORD *)(a1 + 176) != result )
      *(_QWORD *)(a1 + 176) = result;
  }
  return result;
}
