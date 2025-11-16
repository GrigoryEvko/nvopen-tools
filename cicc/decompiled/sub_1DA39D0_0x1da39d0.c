// Function: sub_1DA39D0
// Address: 0x1da39d0
//
__int64 __fastcall sub_1DA39D0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v3; // dl
  unsigned __int64 v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r13
  __int64 v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // r14
  _QWORD *v13; // rdi
  _QWORD *v14; // rcx
  unsigned int v15; // ebx
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *i; // rdx
  _QWORD *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdi
  int v22; // esi
  int v23; // r10d
  _QWORD *v24; // r9
  unsigned int v25; // edx
  _QWORD *v26; // r13
  __int64 v27; // r8
  _QWORD *v28; // r14
  _QWORD *v29; // r15
  __int64 v30; // r12
  __int64 v31; // rax
  _QWORD *v32; // r14
  _QWORD *v33; // rdi
  int v34; // edx
  __int64 v35; // [rsp+10h] [rbp-100h]
  _QWORD *v36; // [rsp+18h] [rbp-F8h]
  _QWORD *v37; // [rsp+18h] [rbp-F8h]
  int v38; // [rsp+20h] [rbp-F0h]
  _QWORD *v39; // [rsp+20h] [rbp-F0h]
  int v41; // [rsp+34h] [rbp-DCh]
  int v42; // [rsp+34h] [rbp-DCh]
  __int64 v43; // [rsp+38h] [rbp-D8h]
  __int64 v44; // [rsp+38h] [rbp-D8h]
  __int64 v45; // [rsp+38h] [rbp-D8h]
  _QWORD v46[26]; // [rsp+40h] [rbp-D0h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v3 = result & 1;
  if ( a2 <= 3 )
  {
    if ( v3 )
      return result;
    v14 = *(_QWORD **)(a1 + 16);
    v15 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
    v36 = v14;
  }
  else
  {
    v4 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v38 = v4;
    v36 = *(_QWORD **)(a1 + 16);
    if ( (unsigned int)v4 > 0x40 )
    {
      v35 = 40LL * (unsigned int)v4;
      if ( v3 )
      {
LABEL_5:
        v5 = (_QWORD *)(a1 + 32);
        v37 = (_QWORD *)(a1 + 192);
        v6 = v46;
        do
        {
          v7 = *(v5 - 2);
          if ( v7 != -8 && v7 != -16 )
          {
            if ( v6 )
              *v6 = v7;
            v8 = (_QWORD *)*v5;
            v6[1] = 0;
            v6[3] = v6 + 2;
            v6[2] = v6 + 2;
            v6[4] = 0;
            if ( v8 == v5 )
            {
              v6[1] = v6 + 2;
              v6 += 5;
            }
            else
            {
              do
              {
                v9 = v8[4];
                v41 = *((_DWORD *)v8 + 4);
                v43 = v8[3];
                v10 = sub_22077B0(40);
                *(_QWORD *)(v10 + 32) = v9;
                *(_QWORD *)(v10 + 24) = v43;
                *(_DWORD *)(v10 + 16) = v41;
                sub_2208C80(v10, v6 + 2);
                ++v6[4];
                v8 = (_QWORD *)*v8;
              }
              while ( v8 != v5 );
              v11 = v6[2];
              v12 = (_QWORD *)*v5;
              v6 += 5;
              *(v6 - 4) = v11;
              while ( v12 != v5 )
              {
                v13 = v12;
                v12 = (_QWORD *)*v12;
                j_j___libc_free_0(v13, 40);
              }
            }
          }
          v5 += 5;
        }
        while ( v37 != v5 );
        *(_BYTE *)(a1 + 8) &= ~1u;
        *(_QWORD *)(a1 + 16) = sub_22077B0(v35);
        *(_DWORD *)(a1 + 24) = v38;
        return (__int64)sub_1DA37A0(a1, v46, v6);
      }
      v15 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      if ( v3 )
      {
        v35 = 2560;
        v38 = 64;
        goto LABEL_5;
      }
      v35 = 2560;
      v38 = 64;
      v15 = *(_DWORD *)(a1 + 24);
    }
    *(_QWORD *)(a1 + 16) = sub_22077B0(v35);
    *(_DWORD *)(a1 + 24) = v38;
  }
  v39 = &v36[5 * v15];
  v44 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = v44 & 1;
  if ( (v44 & 1) != 0 )
  {
    v16 = (_QWORD *)(a1 + 16);
    v17 = 20;
  }
  else
  {
    v16 = *(_QWORD **)(a1 + 16);
    v17 = 5LL * *(unsigned int *)(a1 + 24);
  }
  for ( i = &v16[v17]; i != v16; v16 += 5 )
  {
    if ( v16 )
      *v16 = -8;
  }
  v19 = v36 + 2;
  if ( v39 != v36 )
  {
    while ( 1 )
    {
      v20 = *(v19 - 2);
      if ( v20 != -16 && v20 != -8 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v21 = a1 + 16;
          v22 = 3;
        }
        else
        {
          v34 = *(_DWORD *)(a1 + 24);
          v21 = *(_QWORD *)(a1 + 16);
          if ( !v34 )
          {
            MEMORY[0] = *(v19 - 2);
            BUG();
          }
          v22 = v34 - 1;
        }
        v23 = 1;
        v24 = 0;
        v25 = v22 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v26 = (_QWORD *)(v21 + 40LL * v25);
        v27 = *v26;
        if ( v20 != *v26 )
        {
          while ( v27 != -8 )
          {
            if ( !v24 && v27 == -16 )
              v24 = v26;
            v25 = v22 & (v23 + v25);
            v26 = (_QWORD *)(v21 + 40LL * v25);
            v27 = *v26;
            if ( v20 == *v26 )
              goto LABEL_30;
            ++v23;
          }
          if ( v24 )
            v26 = v24;
        }
LABEL_30:
        v28 = v26 + 2;
        *v26 = v20;
        v26[1] = 0;
        v26[3] = v26 + 2;
        v26[2] = v26 + 2;
        v26[4] = 0;
        v29 = (_QWORD *)*v19;
        if ( v19 != (_QWORD *)*v19 )
        {
          do
          {
            v30 = v29[4];
            v42 = *((_DWORD *)v29 + 4);
            v45 = v29[3];
            v31 = sub_22077B0(40);
            *(_QWORD *)(v31 + 32) = v30;
            *(_DWORD *)(v31 + 16) = v42;
            *(_QWORD *)(v31 + 24) = v45;
            sub_2208C80(v31, v26 + 2);
            ++v26[4];
            v29 = (_QWORD *)*v29;
          }
          while ( v19 != v29 );
          v28 = (_QWORD *)v26[2];
        }
        v26[1] = v28;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v32 = (_QWORD *)*v19;
        while ( v19 != v32 )
        {
          v33 = v32;
          v32 = (_QWORD *)*v32;
          j_j___libc_free_0(v33, 40);
        }
      }
      if ( v39 == v19 + 3 )
        break;
      v19 += 5;
    }
  }
  return j___libc_free_0(v36);
}
