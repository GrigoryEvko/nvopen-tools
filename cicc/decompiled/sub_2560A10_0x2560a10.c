// Function: sub_2560A10
// Address: 0x2560a10
//
void __fastcall sub_2560A10(unsigned int *a1, __int64 a2)
{
  char *v2; // r14
  unsigned __int64 v4; // r8
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rsi
  char *v8; // r9
  __int64 v9; // rax
  _DWORD *v10; // rdx
  char *v11; // rsi
  char *v12; // r12
  __int64 v13; // rbx
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdx
  __int64 *v16; // rbx
  char *v17; // r12
  __int64 v18; // rbx
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  unsigned int v24; // [rsp-44h] [rbp-44h]
  __int64 *v25; // [rsp-40h] [rbp-40h]
  unsigned __int64 v26; // [rsp-40h] [rbp-40h]
  unsigned __int64 v27; // [rsp-40h] [rbp-40h]

  if ( a1 != (unsigned int *)a2 )
  {
    v2 = (char *)(a2 + 16);
    v4 = *(_QWORD *)a1;
    v5 = a1[2];
    v6 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v7 = *(unsigned int *)(a2 + 8);
      v24 = v7;
      if ( v7 <= v5 )
      {
        v15 = *(_QWORD *)a1;
        if ( v7 )
        {
          v22 = v4 + 16 * v7;
          do
          {
            if ( *(_DWORD *)(v4 + 8) > 0x40u && *(_QWORD *)v4 )
            {
              v27 = v4;
              j_j___libc_free_0_0(*(_QWORD *)v4);
              v4 = v27;
            }
            v23 = *(_QWORD *)v2;
            v4 += 16LL;
            v2 += 16;
            *(_QWORD *)(v4 - 16) = v23;
            *(_DWORD *)(v4 - 8) = *((_DWORD *)v2 - 2);
            *((_DWORD *)v2 - 2) = 0;
          }
          while ( v4 != v22 );
          v15 = *(_QWORD *)a1;
          v5 = a1[2];
        }
        v16 = (__int64 *)(v15 + 16 * v5);
        if ( v16 != (__int64 *)v4 )
        {
          do
          {
            v16 -= 2;
            v25 = (__int64 *)v4;
            sub_969240(v16);
            v4 = (unsigned __int64)v25;
          }
          while ( v25 != v16 );
        }
        a1[2] = v7;
        v17 = *(char **)a2;
        v18 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v18 )
        {
          do
          {
            v18 -= 16;
            if ( *(_DWORD *)(v18 + 8) > 0x40u && *(_QWORD *)v18 )
              j_j___libc_free_0_0(*(_QWORD *)v18);
          }
          while ( v17 != (char *)v18 );
        }
      }
      else
      {
        if ( v7 > a1[3] )
        {
          v21 = v4 + 16 * v5;
          while ( v21 != v6 )
          {
            while ( 1 )
            {
              v21 -= 16LL;
              if ( *(_DWORD *)(v21 + 8) <= 0x40u || !*(_QWORD *)v21 )
                break;
              j_j___libc_free_0_0(*(_QWORD *)v21);
              if ( v21 == v6 )
                goto LABEL_46;
            }
          }
LABEL_46:
          a1[2] = 0;
          v5 = 0;
          sub_AE4800(a1, v7);
          v2 = *(char **)a2;
          v7 = *(unsigned int *)(a2 + 8);
          v6 = *(_QWORD *)a1;
          v8 = *(char **)a2;
        }
        else
        {
          v8 = v2;
          if ( a1[2] )
          {
            v5 *= 16LL;
            v19 = v4 + v5;
            do
            {
              if ( *(_DWORD *)(v4 + 8) > 0x40u && *(_QWORD *)v4 )
              {
                v26 = v4;
                j_j___libc_free_0_0(*(_QWORD *)v4);
                v4 = v26;
              }
              v20 = *(_QWORD *)v2;
              v4 += 16LL;
              v2 += 16;
              *(_QWORD *)(v4 - 16) = v20;
              *(_DWORD *)(v4 - 8) = *((_DWORD *)v2 - 2);
              *((_DWORD *)v2 - 2) = 0;
            }
            while ( v4 != v19 );
            v2 = *(char **)a2;
            v7 = *(unsigned int *)(a2 + 8);
            v6 = *(_QWORD *)a1;
            v8 = (char *)(*(_QWORD *)a2 + v5);
          }
        }
        v9 = v6 + v5;
        v10 = v8 + 8;
        v11 = &v2[16 * v7];
        if ( v8 != v11 )
        {
          do
          {
            if ( v9 )
            {
              *(_DWORD *)(v9 + 8) = *v10;
              *(_QWORD *)v9 = *((_QWORD *)v10 - 1);
              *v10 = 0;
            }
            v9 += 16;
            v10 += 4;
          }
          while ( v9 != v6 + v5 + v11 - v8 );
        }
        a1[2] = v24;
        v12 = *(char **)a2;
        v13 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v13 )
        {
          do
          {
            v13 -= 16;
            if ( *(_DWORD *)(v13 + 8) > 0x40u )
            {
              if ( *(_QWORD *)v13 )
                j_j___libc_free_0_0(*(_QWORD *)v13);
            }
          }
          while ( v12 != (char *)v13 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v14 = v4 + 16 * v5;
      if ( v14 != v4 )
      {
        do
        {
          v14 -= 16LL;
          if ( *(_DWORD *)(v14 + 8) > 0x40u && *(_QWORD *)v14 )
            j_j___libc_free_0_0(*(_QWORD *)v14);
        }
        while ( v14 != v6 );
        v4 = *(_QWORD *)a1;
      }
      if ( (unsigned int *)v4 != a1 + 4 )
        _libc_free(v4);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      a1[2] = *(_DWORD *)(a2 + 8);
      a1[3] = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v2;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
