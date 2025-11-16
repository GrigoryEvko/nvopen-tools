// Function: sub_37B58E0
// Address: 0x37b58e0
//
__int64 __fastcall sub_37B58E0(__int64 a1, __int64 a2)
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
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // r15
  __int64 (__fastcall *v24)(__int64, unsigned __int16); // rax
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // r15
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // r8
  __int64 (__fastcall *v32)(__int64, unsigned __int16); // rax
  bool v33; // cc
  __int64 v34; // rax
  _QWORD *v35; // rax
  _QWORD *i; // rsi
  unsigned __int64 v37; // rdx
  __int16 v38; // cx
  unsigned int v39; // edx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // [rsp+0h] [rbp-40h]
  __int64 v44; // [rsp+8h] [rbp-38h]
  unsigned int v45; // [rsp+8h] [rbp-38h]
  __int64 v46; // [rsp+8h] [rbp-38h]

  if ( a2 )
  {
    v3 = *(_QWORD *)a2;
    if ( *(int *)(*(_QWORD *)a2 + 24LL) < 0 )
    {
      v18 = 0;
      v19 = *(unsigned int *)(v3 + 68);
      v20 = 16 * v19;
      if ( (_DWORD)v19 )
      {
        do
        {
          v21 = *(unsigned __int16 *)(*(_QWORD *)(v3 + 48) + v18);
          if ( (_WORD)v21 )
          {
            v22 = *(_QWORD *)(a1 + 136);
            v23 = *(_QWORD *)(v22 + 8LL * (unsigned __int16)v21 + 112);
            if ( v23 )
            {
              v24 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v22 + 552LL);
              if ( v24 == sub_2EC09E0
                || (v46 = v20,
                    v40 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v24)(v22, v21, 0),
                    v20 = v46,
                    (v23 = v40) != 0) )
              {
                v44 = v20;
                v25 = sub_37B40C0(a1, a2, *(unsigned __int16 *)(*(_QWORD *)v23 + 24LL));
                v20 = v44;
                *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * *(unsigned __int16 *)(*(_QWORD *)v23 + 24LL)) += v25;
              }
            }
          }
          v18 += 16;
        }
        while ( v20 != v18 );
      }
      v26 = *(unsigned int *)(v3 + 64);
      if ( (_DWORD)v26 )
      {
        v27 = 0;
        v28 = 40 * v26;
        do
        {
          v29 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v27 + *(_QWORD *)(v3 + 40)) + 48LL)
                                    + 16LL * *(unsigned int *)(v27 + *(_QWORD *)(v3 + 40) + 8));
          if ( (_WORD)v29 )
          {
            v30 = *(_QWORD *)(a1 + 136);
            v31 = *(_QWORD *)(v30 + 8LL * (unsigned __int16)v29 + 112);
            if ( v31 )
            {
              v32 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v30 + 552LL);
              if ( v32 == sub_2EC09E0
                || (v31 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v32)(v30, v29, 0)) != 0 )
              {
                v43 = v31;
                v45 = *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * *(unsigned __int16 *)(*(_QWORD *)v31 + 24LL));
                v33 = v45 <= (unsigned int)sub_37B3FA0(a1, a2, *(unsigned __int16 *)(*(_QWORD *)v31 + 24LL));
                v34 = *(_QWORD *)v43;
                if ( v33 )
                  *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * *(unsigned __int16 *)(v34 + 24)) = 0;
                else
                  *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * *(unsigned __int16 *)(*(_QWORD *)v43 + 24LL)) -= sub_37B3FA0(a1, a2, *(unsigned __int16 *)(v34 + 24));
              }
            }
          }
          v27 += 40;
        }
        while ( v28 != v27 );
      }
      v35 = *(_QWORD **)(a2 + 40);
      for ( i = &v35[2 * *(unsigned int *)(a2 + 48)]; i != v35; v35 += 2 )
      {
        if ( (*v35 & 6) == 0 )
        {
          v37 = *v35 & 0xFFFFFFFFFFFFFFF8LL;
          v38 = *(_WORD *)(v37 + 250);
          if ( v38 )
            *(_WORD *)(v37 + 250) = v38 - 1;
        }
      }
    }
    v5 = 0;
    sub_37B5710(a1, (__int64 *)a2);
    v6 = *(__int64 **)(a2 + 120);
    v7 = &v6[2 * *(unsigned int *)(a2 + 128)];
    if ( v7 == v6 )
      goto LABEL_33;
    do
    {
      sub_37B4C50(a1, *v6 & 0xFFFFFFFFFFFFFFF8LL);
      v5 += ((*v6 >> 1) & 3) == 0;
      v6 += 2;
    }
    while ( v7 != v6 );
    if ( !v5 )
    {
LABEL_33:
      v39 = *(_DWORD *)(a2 + 208);
      result = 0;
      if ( *(_DWORD *)(a1 + 192) >= v39 )
        result = *(_DWORD *)(a1 + 192) - v39;
      *(_DWORD *)(a1 + 192) = result;
    }
    else
    {
      result = *(unsigned __int16 *)(a2 + 250);
      *(_DWORD *)(a1 + 192) += result;
    }
    v9 = *(__int64 **)(a2 + 120);
    v10 = 0;
    v11 = *(unsigned int *)(a2 + 128);
    for ( j = &v9[2 * v11]; j != v9; v9 += 2 )
    {
      result = (*v9 >> 1) & 3;
      v10 -= (result == 0) - 1;
    }
    v13 = *(_DWORD *)(a1 + 196) + v11 - v10;
    v14 = 0;
    *(_DWORD *)(a1 + 196) = v13;
    v15 = *(__int64 **)(a2 + 40);
    v16 = *(unsigned int *)(a2 + 48);
    for ( k = &v15[2 * v16]; k != v15; v15 += 2 )
    {
      result = (*v15 >> 1) & 3;
      v14 -= (result == 0) - 1;
    }
    *(_DWORD *)(a1 + 196) = v13 - v16 + v14;
  }
  else
  {
    v41 = *(_QWORD *)(a1 + 160);
    v42 = *(_QWORD *)(v41 + 24);
    *(_QWORD *)(v41 + 40) = 1;
    if ( v42 )
      sub_37B5390(v42);
    result = *(_QWORD *)(a1 + 168);
    if ( *(_QWORD *)(a1 + 176) != result )
      *(_QWORD *)(a1 + 176) = result;
  }
  return result;
}
