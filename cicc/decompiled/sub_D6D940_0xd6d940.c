// Function: sub_D6D940
// Address: 0xd6d940
//
unsigned __int64 __fastcall sub_D6D940(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  unsigned __int64 result; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // rcx
  unsigned __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdi
  int v26; // eax
  int v27; // edx
  __int64 v28; // rdx
  __int64 v29; // r9
  __int64 *v30; // rax
  __int64 v31; // r8
  __int64 v32; // r10
  __int64 v33; // rdi

  v6 = *a1;
  result = sub_D68B40(*a1, a2);
  if ( result )
  {
    v8 = result;
    v9 = sub_10420D0(v6, a4);
    v10 = *(_QWORD *)(v8 - 8);
    v11 = v9;
    if ( (*(_DWORD *)(v8 + 4) & 0x7FFFFFF) == 0 )
      goto LABEL_31;
    v12 = 0;
    v13 = 8LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
    do
    {
      v14 = *(_QWORD *)(v10 + 32LL * *(unsigned int *)(v8 + 76) + v12);
      if ( a3 != v14 )
      {
        sub_D689D0(v11, *(_QWORD *)(v10 + 4 * v12), v14);
        v10 = *(_QWORD *)(v8 - 8);
      }
      v12 += 8;
    }
    while ( v13 != v12 );
    if ( (*(_DWORD *)(v8 + 4) & 0x7FFFFFF) != 0 )
    {
      v15 = 0;
      while ( a3 != *(_QWORD *)(v10 + 32LL * *(unsigned int *)(v8 + 76) + 8 * v15) )
      {
        if ( (*(_DWORD *)(v8 + 4) & 0x7FFFFFF) == (_DWORD)++v15 )
          goto LABEL_31;
      }
      v16 = 32 * v15;
    }
    else
    {
LABEL_31:
      v16 = 0x1FFFFFFFE0LL;
    }
    sub_AC2B30(v10, *(_QWORD *)(v10 + v16));
    *(_QWORD *)(*(_QWORD *)(v8 - 8) + 32LL * *(unsigned int *)(v8 + 76)) = a3;
    v17 = (*(_DWORD *)(v8 + 4) & 0x7FFFFFF) - 1;
    if ( (*(_DWORD *)(v8 + 4) & 0x7FFFFFF) != 1 )
    {
      v18 = 8LL * v17;
      v19 = 8 * (v17 - (unsigned __int64)((*(_DWORD *)(v8 + 4) & 0x7FFFFFFu) - 2));
      while ( 1 )
      {
        v28 = v17;
        v22 = *(_QWORD *)(v8 - 8);
        v29 = 32LL * v17;
        v30 = (__int64 *)(v22 + 4 * v18);
        v31 = *(_QWORD *)(v22 + v29);
        v32 = *v30;
        if ( v31 )
        {
          if ( v32 )
          {
            v20 = v30[1];
            *(_QWORD *)v30[2] = v20;
            if ( v20 )
              *(_QWORD *)(v20 + 16) = v30[2];
          }
          *v30 = v31;
          v21 = *(_QWORD *)(v31 + 16);
          v30[1] = v21;
          if ( v21 )
            *(_QWORD *)(v21 + 16) = v30 + 1;
          v30[2] = v31 + 16;
          *(_QWORD *)(v31 + 16) = v30;
          v22 = *(_QWORD *)(v8 - 8);
        }
        else if ( v32 )
        {
          v33 = v30[1];
          *(_QWORD *)v30[2] = v33;
          if ( v33 )
            *(_QWORD *)(v33 + 16) = v30[2];
          *v30 = 0;
          v22 = *(_QWORD *)(v8 - 8);
        }
        v23 = 8 * v28;
        *(_QWORD *)(v18 + v22 + 32LL * *(unsigned int *)(v8 + 76)) = *(_QWORD *)(v22
                                                                               + 32LL * *(unsigned int *)(v8 + 76)
                                                                               + v23);
        v24 = v29 + *(_QWORD *)(v8 - 8);
        if ( *(_QWORD *)v24 )
        {
          v25 = *(_QWORD *)(v24 + 8);
          **(_QWORD **)(v24 + 16) = v25;
          if ( v25 )
            *(_QWORD *)(v25 + 16) = *(_QWORD *)(v24 + 16);
        }
        *(_QWORD *)v24 = 0;
        *(_QWORD *)(*(_QWORD *)(v8 - 8) + 32LL * *(unsigned int *)(v8 + 76) + v23) = 0;
        v26 = *(_DWORD *)(v8 + 4);
        v27 = (v26 + 0x7FFFFFF) & 0x7FFFFFF;
        *(_DWORD *)(v8 + 4) = v27 | v26 & 0xF8000000;
        if ( v18 == v19 )
          break;
        v18 -= 8;
        v17 = v27 - 1;
      }
    }
    sub_D689D0(v8, v11, a4);
    return sub_D6D630((__int64)a1, v11);
  }
  return result;
}
