// Function: sub_CF9EC0
// Address: 0xcf9ec0
//
__int64 __fastcall sub_CF9EC0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // r12
  __int64 i; // rdx
  __int64 v10; // rbx
  __int64 v11; // r15
  int v12; // r13d
  unsigned int v13; // eax
  int v14; // r10d
  __int64 *v15; // r9
  unsigned __int64 v16; // rax
  unsigned int j; // r8d
  __int64 *v18; // rax
  __int64 v19; // rdi
  int v20; // edx
  __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned int v23; // r8d
  __int64 v24; // rdx
  __int64 k; // rdx
  __int64 v26; // [rsp+0h] [rbp-60h]
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  unsigned int v29; // [rsp+1Ch] [rbp-44h]
  unsigned int v30[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = sub_C7D670(48LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v27 = 48 * v4;
    v8 = v5 + 48 * v4;
    for ( i = result + 48LL * *(unsigned int *)(a1 + 24); i != result; result += 48 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = 100;
      }
    }
    if ( v8 != v5 )
    {
      v26 = v5;
      v10 = v5;
      while ( 1 )
      {
        v11 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 == -4096 )
        {
          if ( *(_DWORD *)(v10 + 8) != 100 )
            goto LABEL_12;
          v10 += 48;
          if ( v8 == v10 )
            goto LABEL_23;
        }
        else if ( v11 == -8192 && *(_DWORD *)(v10 + 8) == 101 )
        {
          v10 += 48;
          if ( v8 == v10 )
            goto LABEL_23;
        }
        else
        {
LABEL_12:
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(_QWORD *)v10;
            BUG();
          }
          v28 = *(_QWORD *)(a1 + 8);
          v30[0] = *(_DWORD *)(v10 + 8);
          v29 = v30[0];
          v13 = sub_CF97C0(v30);
          v14 = 1;
          v15 = 0;
          v16 = 0xBF58476D1CE4E5B9LL
              * (v13 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32));
          for ( j = (v12 - 1) & ((v16 >> 31) ^ v16); ; j = (v12 - 1) & v23 )
          {
            v18 = (__int64 *)(v28 + 48LL * j);
            v19 = *v18;
            if ( v11 == *v18 && *((_DWORD *)v18 + 2) == v29 )
              break;
            if ( v19 == -4096 )
            {
              if ( *((_DWORD *)v18 + 2) == 100 )
              {
                if ( v15 )
                  v18 = v15;
                break;
              }
            }
            else if ( v19 == -8192 && *((_DWORD *)v18 + 2) == 101 && !v15 )
            {
              v15 = (__int64 *)(v28 + 48LL * j);
            }
            v23 = v14 + j;
            ++v14;
          }
          *v18 = v11;
          v20 = *(_DWORD *)(v10 + 8);
          v18[4] = 0;
          v18[3] = 0;
          *((_DWORD *)v18 + 10) = 0;
          *((_DWORD *)v18 + 2) = v20;
          v18[2] = 1;
          v21 = *(_QWORD *)(v10 + 24);
          ++*(_QWORD *)(v10 + 16);
          v22 = v18[3];
          v10 += 48;
          v18[3] = v21;
          LODWORD(v21) = *(_DWORD *)(v10 - 16);
          *(_QWORD *)(v10 - 24) = v22;
          LODWORD(v22) = *((_DWORD *)v18 + 8);
          *((_DWORD *)v18 + 8) = v21;
          LODWORD(v21) = *(_DWORD *)(v10 - 12);
          *(_DWORD *)(v10 - 16) = v22;
          LODWORD(v22) = *((_DWORD *)v18 + 9);
          *((_DWORD *)v18 + 9) = v21;
          LODWORD(v21) = *(_DWORD *)(v10 - 8);
          *(_DWORD *)(v10 - 12) = v22;
          LODWORD(v22) = *((_DWORD *)v18 + 10);
          *((_DWORD *)v18 + 10) = v21;
          *(_DWORD *)(v10 - 8) = v22;
          ++*(_DWORD *)(a1 + 16);
          sub_C7D6A0(*(_QWORD *)(v10 - 24), 24LL * *(unsigned int *)(v10 - 8), 8);
          if ( v8 == v10 )
          {
LABEL_23:
            v5 = v26;
            return sub_C7D6A0(v5, v27, 8);
          }
        }
      }
    }
    return sub_C7D6A0(v5, v27, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 48 * v24; k != result; result += 48 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = 100;
      }
    }
  }
  return result;
}
