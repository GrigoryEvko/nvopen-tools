// Function: sub_27578C0
// Address: 0x27578c0
//
void __fastcall sub_27578C0(__int64 a1, __int64 *a2)
{
  __int64 *v2; // r14
  unsigned __int64 v5; // r8
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rsi
  int v9; // r13d
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rsi
  unsigned __int64 v13; // rdi
  int v14; // ecx
  __int64 *v15; // r13
  __int64 v16; // r12
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r14
  unsigned __int64 v22; // rdi
  __int64 *v23; // r13
  __int64 v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v26; // r15
  unsigned __int64 v27; // rdi
  __int64 v28; // rax
  unsigned __int64 v29; // r14
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // r15
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  unsigned __int64 v34; // [rsp-48h] [rbp-48h]
  unsigned __int64 v35; // [rsp-48h] [rbp-48h]
  unsigned __int64 v36; // [rsp-40h] [rbp-40h]
  unsigned __int64 v37; // [rsp-40h] [rbp-40h]
  unsigned __int64 v38; // [rsp-40h] [rbp-40h]
  unsigned __int64 v39; // [rsp-40h] [rbp-40h]
  unsigned __int64 v40; // [rsp-40h] [rbp-40h]
  unsigned __int64 v41; // [rsp-40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v2 = a2 + 2;
    v5 = *(_QWORD *)a1;
    v6 = *(unsigned int *)(a1 + 8);
    v7 = *(_QWORD *)a1;
    if ( (__int64 *)*a2 == a2 + 2 )
    {
      v8 = *((unsigned int *)a2 + 2);
      v9 = v8;
      if ( v8 <= v6 )
      {
        v20 = *(_QWORD *)a1;
        if ( v8 )
        {
          v31 = v5 + 32 * v8;
          do
          {
            if ( *(_DWORD *)(v5 + 8) > 0x40u && *(_QWORD *)v5 )
            {
              v40 = v5;
              j_j___libc_free_0_0(*(_QWORD *)v5);
              v5 = v40;
            }
            *(_QWORD *)v5 = *v2;
            *(_DWORD *)(v5 + 8) = *((_DWORD *)v2 + 2);
            *((_DWORD *)v2 + 2) = 0;
            if ( *(_DWORD *)(v5 + 24) > 0x40u )
            {
              v32 = *(_QWORD *)(v5 + 16);
              if ( v32 )
              {
                v41 = v5;
                j_j___libc_free_0_0(v32);
                v5 = v41;
              }
            }
            v33 = v2[2];
            v5 += 32LL;
            v2 += 4;
            *(_QWORD *)(v5 - 16) = v33;
            *(_DWORD *)(v5 - 8) = *((_DWORD *)v2 - 2);
            *((_DWORD *)v2 - 2) = 0;
          }
          while ( v5 != v31 );
          v20 = *(_QWORD *)a1;
          v6 = *(unsigned int *)(a1 + 8);
        }
        v21 = v20 + 32 * v6;
        while ( v5 != v21 )
        {
          v21 -= 32LL;
          if ( *(_DWORD *)(v21 + 24) > 0x40u )
          {
            v22 = *(_QWORD *)(v21 + 16);
            if ( v22 )
            {
              v36 = v5;
              j_j___libc_free_0_0(v22);
              v5 = v36;
            }
          }
          if ( *(_DWORD *)(v21 + 8) > 0x40u && *(_QWORD *)v21 )
          {
            v37 = v5;
            j_j___libc_free_0_0(*(_QWORD *)v21);
            v5 = v37;
          }
        }
        *(_DWORD *)(a1 + 8) = v8;
        v23 = (__int64 *)*a2;
        v24 = *a2 + 32LL * *((unsigned int *)a2 + 2);
        if ( *a2 != v24 )
        {
          do
          {
            v24 -= 32;
            if ( *(_DWORD *)(v24 + 24) > 0x40u )
            {
              v25 = *(_QWORD *)(v24 + 16);
              if ( v25 )
                j_j___libc_free_0_0(v25);
            }
            if ( *(_DWORD *)(v24 + 8) > 0x40u && *(_QWORD *)v24 )
              j_j___libc_free_0_0(*(_QWORD *)v24);
          }
          while ( v23 != (__int64 *)v24 );
        }
      }
      else
      {
        if ( v8 > *(unsigned int *)(a1 + 12) )
        {
          v29 = v5 + 32 * v6;
          while ( v7 != v29 )
          {
            while ( 1 )
            {
              v29 -= 32LL;
              if ( *(_DWORD *)(v29 + 24) > 0x40u )
              {
                v30 = *(_QWORD *)(v29 + 16);
                if ( v30 )
                  j_j___libc_free_0_0(v30);
              }
              if ( *(_DWORD *)(v29 + 8) <= 0x40u || !*(_QWORD *)v29 )
                break;
              j_j___libc_free_0_0(*(_QWORD *)v29);
              if ( v7 == v29 )
                goto LABEL_67;
            }
          }
LABEL_67:
          *(_DWORD *)(a1 + 8) = 0;
          sub_9D5330(a1, v8);
          v2 = (__int64 *)*a2;
          v8 = *((unsigned int *)a2 + 2);
          v6 = 0;
          v7 = *(_QWORD *)a1;
          v10 = *a2;
        }
        else
        {
          v10 = (__int64)v2;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v6 *= 32LL;
            v26 = v5 + v6;
            do
            {
              if ( *(_DWORD *)(v5 + 8) > 0x40u && *(_QWORD *)v5 )
              {
                v34 = v6;
                v38 = v5;
                j_j___libc_free_0_0(*(_QWORD *)v5);
                v6 = v34;
                v5 = v38;
              }
              *(_QWORD *)v5 = *v2;
              *(_DWORD *)(v5 + 8) = *((_DWORD *)v2 + 2);
              *((_DWORD *)v2 + 2) = 0;
              if ( *(_DWORD *)(v5 + 24) > 0x40u )
              {
                v27 = *(_QWORD *)(v5 + 16);
                if ( v27 )
                {
                  v35 = v6;
                  v39 = v5;
                  j_j___libc_free_0_0(v27);
                  v6 = v35;
                  v5 = v39;
                }
              }
              v28 = v2[2];
              v5 += 32LL;
              v2 += 4;
              *(_QWORD *)(v5 - 16) = v28;
              *(_DWORD *)(v5 - 8) = *((_DWORD *)v2 - 2);
              *((_DWORD *)v2 - 2) = 0;
            }
            while ( v5 != v26 );
            v2 = (__int64 *)*a2;
            v8 = *((unsigned int *)a2 + 2);
            v7 = *(_QWORD *)a1;
            v10 = *a2 + v6;
          }
        }
        v11 = v7 + v6;
        v12 = &v2[4 * v8];
        v13 = (unsigned __int64)v12 + v7 + v6 - v10;
        if ( v12 != (__int64 *)v10 )
        {
          do
          {
            if ( v11 )
            {
              *(_DWORD *)(v11 + 8) = *(_DWORD *)(v10 + 8);
              *(_QWORD *)v11 = *(_QWORD *)v10;
              v14 = *(_DWORD *)(v10 + 24);
              *(_DWORD *)(v10 + 8) = 0;
              *(_DWORD *)(v11 + 24) = v14;
              *(_QWORD *)(v11 + 16) = *(_QWORD *)(v10 + 16);
              *(_DWORD *)(v10 + 24) = 0;
            }
            v11 += 32;
            v10 += 32;
          }
          while ( v11 != v13 );
        }
        *(_DWORD *)(a1 + 8) = v9;
        v15 = (__int64 *)*a2;
        v16 = *a2 + 32LL * *((unsigned int *)a2 + 2);
        if ( *a2 != v16 )
        {
          do
          {
            v16 -= 32;
            if ( *(_DWORD *)(v16 + 24) > 0x40u )
            {
              v17 = *(_QWORD *)(v16 + 16);
              if ( v17 )
                j_j___libc_free_0_0(v17);
            }
            if ( *(_DWORD *)(v16 + 8) > 0x40u )
            {
              if ( *(_QWORD *)v16 )
                j_j___libc_free_0_0(*(_QWORD *)v16);
            }
          }
          while ( v15 != (__int64 *)v16 );
        }
      }
      *((_DWORD *)a2 + 2) = 0;
    }
    else
    {
      v18 = v5 + 32 * v6;
      if ( v18 != v5 )
      {
        do
        {
          v18 -= 32LL;
          if ( *(_DWORD *)(v18 + 24) > 0x40u )
          {
            v19 = *(_QWORD *)(v18 + 16);
            if ( v19 )
              j_j___libc_free_0_0(v19);
          }
          if ( *(_DWORD *)(v18 + 8) > 0x40u && *(_QWORD *)v18 )
            j_j___libc_free_0_0(*(_QWORD *)v18);
        }
        while ( v18 != v7 );
        v5 = *(_QWORD *)a1;
      }
      if ( v5 != a1 + 16 )
        _libc_free(v5);
      *(_QWORD *)a1 = *a2;
      *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
      *(_DWORD *)(a1 + 12) = *((_DWORD *)a2 + 3);
      *a2 = (__int64)v2;
      a2[1] = 0;
    }
  }
}
