// Function: sub_2D267E0
// Address: 0x2d267e0
//
_QWORD *__fastcall sub_2D267E0(__int64 a1, int a2)
{
  __int64 v2; // r15
  __int64 v3; // rbx
  unsigned int v4; // eax
  _QWORD *result; // rax
  _QWORD *i; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r12
  int v15; // edx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  unsigned __int64 v21; // r12
  __int64 v22; // rax
  unsigned __int64 v23; // r13
  unsigned __int64 v24; // r15
  __int64 v25; // rsi
  __int64 v26; // r13
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 *v29; // rax
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // r12
  __int64 v37; // rax
  unsigned __int64 v38; // r15
  unsigned __int64 v39; // r14
  __int64 v40; // rsi
  _QWORD *j; // rdx
  __int64 v42; // [rsp+0h] [rbp-70h]
  unsigned int v43; // [rsp+Ch] [rbp-64h]
  __int64 v44; // [rsp+10h] [rbp-60h]
  unsigned int v45; // [rsp+18h] [rbp-58h]
  __int64 *v46; // [rsp+18h] [rbp-58h]
  __int64 v47; // [rsp+18h] [rbp-58h]
  __int64 v48; // [rsp+20h] [rbp-50h]
  __int64 v50; // [rsp+30h] [rbp-40h]
  __int64 v51; // [rsp+38h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(unsigned int *)(a1 + 24);
  v48 = v2;
  v4 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_C7D670(56LL * v4, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v2 )
  {
    v44 = 56 * v3;
    v50 = 56 * v3 + v2;
    *(_QWORD *)(a1 + 16) = 0;
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    v7 = v2 + 56;
    if ( v50 != v2 )
    {
      do
      {
        v8 = *(_QWORD *)(v7 - 56);
        v51 = v7;
        if ( v8 != -8192 && v8 != -4096 )
        {
          v9 = *(_DWORD *)(a1 + 24);
          if ( !v9 )
          {
            MEMORY[0] = *(_QWORD *)(v7 - 56);
            BUG();
          }
          v10 = *(_QWORD *)(a1 + 8);
          v11 = (unsigned int)(v9 - 1);
          v12 = 0;
          v13 = (unsigned int)v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v14 = v10 + 56 * v13;
          v15 = 1;
          v16 = *(_QWORD *)v14;
          if ( v8 != *(_QWORD *)v14 )
          {
            while ( v16 != -4096 )
            {
              if ( !v12 && v16 == -8192 )
                v12 = v14;
              v13 = (unsigned int)v11 & ((_DWORD)v13 + v15);
              v14 = v10 + 56 * v13;
              v16 = *(_QWORD *)v14;
              if ( v8 == *(_QWORD *)v14 )
                goto LABEL_13;
              ++v15;
            }
            if ( v12 )
              v14 = v12;
          }
LABEL_13:
          *(_QWORD *)(v14 + 24) = 0;
          *(_QWORD *)(v14 + 16) = 0;
          *(_DWORD *)(v14 + 32) = 0;
          *(_QWORD *)v14 = v8;
          *(_QWORD *)(v14 + 8) = 1;
          v17 = *(_QWORD *)(v7 - 40);
          ++*(_QWORD *)(v7 - 48);
          v18 = *(_QWORD *)(v14 + 16);
          *(_QWORD *)(v14 + 16) = v17;
          LODWORD(v17) = *(_DWORD *)(v7 - 32);
          *(_QWORD *)(v7 - 40) = v18;
          LODWORD(v18) = *(_DWORD *)(v14 + 24);
          *(_DWORD *)(v14 + 24) = v17;
          LODWORD(v17) = *(_DWORD *)(v7 - 28);
          *(_DWORD *)(v7 - 32) = v18;
          LODWORD(v18) = *(_DWORD *)(v14 + 28);
          *(_DWORD *)(v14 + 28) = v17;
          LODWORD(v17) = *(_DWORD *)(v7 - 24);
          *(_DWORD *)(v7 - 28) = v18;
          LODWORD(v18) = *(_DWORD *)(v14 + 32);
          *(_DWORD *)(v14 + 32) = v17;
          *(_DWORD *)(v7 - 24) = v18;
          *(_QWORD *)(v14 + 40) = v14 + 56;
          *(_QWORD *)(v14 + 48) = 0;
          v19 = *(unsigned int *)(v7 - 8);
          if ( (_DWORD)v19 && v14 + 40 != v7 - 16 )
          {
            v26 = *(_QWORD *)(v7 - 16);
            if ( v7 == v26 )
            {
              v45 = *(_DWORD *)(v7 - 8);
              sub_2D26690(v14 + 40, (unsigned int)v19, v19, v11, v13, v12);
              v29 = *(__int64 **)(v7 - 16);
              v30 = *(_QWORD *)(v14 + 40);
              v31 = v45;
              v32 = (__int64)&v29[9 * *(unsigned int *)(v7 - 8)];
              if ( v29 != (__int64 *)v32 )
              {
                do
                {
                  while ( 1 )
                  {
                    if ( v30 )
                    {
                      v33 = *v29;
                      *(_DWORD *)(v30 + 16) = 0;
                      *(_DWORD *)(v30 + 20) = 2;
                      *(_QWORD *)v30 = v33;
                      *(_QWORD *)(v30 + 8) = v30 + 24;
                      if ( *((_DWORD *)v29 + 4) )
                        break;
                    }
                    v29 += 9;
                    v30 += 72;
                    if ( (__int64 *)v32 == v29 )
                      goto LABEL_37;
                  }
                  v34 = v30 + 8;
                  v42 = v32;
                  v30 += 72;
                  v43 = v31;
                  v46 = v29;
                  sub_2D262B0(v34, (__int64)(v29 + 1), v31, v32, v27, v28);
                  v32 = v42;
                  v31 = v43;
                  v29 = v46 + 9;
                }
                while ( (__int64 *)v42 != v46 + 9 );
              }
LABEL_37:
              *(_DWORD *)(v14 + 48) = v31;
              v35 = *(_QWORD *)(v26 - 16);
              v36 = v35 + 72LL * *(unsigned int *)(v26 - 8);
              v47 = v35;
              while ( v47 != v36 )
              {
                v37 = *(unsigned int *)(v36 - 56);
                v38 = *(_QWORD *)(v36 - 64);
                v36 -= 72;
                v39 = v38 + 24 * v37;
                if ( v38 != v39 )
                {
                  do
                  {
                    v40 = *(_QWORD *)(v39 - 8);
                    v39 -= 24LL;
                    if ( v40 )
                      sub_B91220(v39 + 16, v40);
                  }
                  while ( v38 != v39 );
                  v38 = *(_QWORD *)(v36 + 8);
                }
                if ( v38 != v36 + 24 )
                  _libc_free(v38);
              }
              *(_DWORD *)(v26 - 8) = 0;
            }
            else
            {
              *(_QWORD *)(v14 + 40) = v26;
              *(_DWORD *)(v14 + 48) = *(_DWORD *)(v7 - 8);
              *(_DWORD *)(v14 + 52) = *(_DWORD *)(v7 - 4);
              *(_QWORD *)(v7 - 16) = v7;
              *(_DWORD *)(v7 - 4) = 0;
              *(_DWORD *)(v7 - 8) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v20 = *(_QWORD *)(v7 - 16);
          v21 = v20 + 72LL * *(unsigned int *)(v7 - 8);
          if ( v20 != v21 )
          {
            do
            {
              v22 = *(unsigned int *)(v21 - 56);
              v23 = *(_QWORD *)(v21 - 64);
              v21 -= 72LL;
              v24 = v23 + 24 * v22;
              if ( v23 != v24 )
              {
                do
                {
                  v25 = *(_QWORD *)(v24 - 8);
                  v24 -= 24LL;
                  if ( v25 )
                    sub_B91220(v24 + 16, v25);
                }
                while ( v23 != v24 );
                v23 = *(_QWORD *)(v21 + 8);
              }
              if ( v23 != v21 + 24 )
                _libc_free(v23);
            }
            while ( v20 != v21 );
            v21 = *(_QWORD *)(v7 - 16);
          }
          if ( v7 != v21 )
            _libc_free(v21);
          sub_C7D6A0(*(_QWORD *)(v7 - 40), 16LL * *(unsigned int *)(v7 - 24), 8);
        }
        v7 += 56;
      }
      while ( v51 != v50 );
    }
    return (_QWORD *)sub_C7D6A0(v48, v44, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * *(unsigned int *)(a1 + 24)]; j != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
