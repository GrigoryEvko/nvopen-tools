// Function: sub_1996030
// Address: 0x1996030
//
void __fastcall sub_1996030(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rsi
  int v12; // r15d
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned __int64 v19; // r12
  __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // r15
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rbx
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // r12
  __int64 v28; // rbx
  unsigned __int64 v29; // rdi
  __int64 v30; // r12
  char **v31; // rdx
  __int64 v32; // rdi
  unsigned __int64 v33; // rbx
  unsigned __int64 v34; // rdi
  __int64 v35; // rbx
  char **v36; // rcx
  __int64 v37; // rdi
  unsigned __int64 v38; // [rsp-48h] [rbp-48h]
  unsigned __int64 v39; // [rsp-40h] [rbp-40h]
  unsigned __int64 v40; // [rsp-40h] [rbp-40h]
  unsigned __int64 v41; // [rsp-40h] [rbp-40h]
  char **v42; // [rsp-40h] [rbp-40h]
  char **v43; // [rsp-40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v8 = *(_QWORD *)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v39 = (unsigned __int64)(a2 + 2);
    v10 = *(_QWORD *)a1;
    if ( (__int64 *)*a2 == a2 + 2 )
    {
      v11 = *((unsigned int *)a2 + 2);
      v12 = v11;
      if ( v11 <= v9 )
      {
        v24 = *(_QWORD *)a1;
        if ( v11 )
        {
          v35 = v8 + 32;
          v36 = (char **)(a2 + 6);
          do
          {
            v37 = v35;
            v43 = v36;
            v35 += 96;
            *(_QWORD *)(v35 - 128) = *(v36 - 4);
            *(_QWORD *)(v35 - 120) = *(v36 - 3);
            *(_BYTE *)(v35 - 112) = *((_BYTE *)v36 - 16);
            *(_QWORD *)(v35 - 104) = *(v36 - 1);
            sub_19931B0(v37, v36, v24, (__int64)v36, v10, a6);
            v36 = v43 + 12;
            *(_QWORD *)(v35 - 48) = v43[6];
            *(_QWORD *)(v35 - 40) = v43[7];
          }
          while ( v8 + 32 + 96 * v11 != v35 );
          v24 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
          v10 = v8 + 96 * v11;
        }
        v25 = v24 + 96 * v9;
        while ( v10 != v25 )
        {
          v25 -= 96;
          v26 = *(_QWORD *)(v25 + 32);
          if ( v26 != v25 + 48 )
          {
            v41 = v10;
            _libc_free(v26);
            v10 = v41;
          }
        }
        *(_DWORD *)(a1 + 8) = v11;
        v27 = *a2;
        v28 = *a2 + 96LL * *((unsigned int *)a2 + 2);
        if ( *a2 != v28 )
        {
          do
          {
            v28 -= 96;
            v29 = *(_QWORD *)(v28 + 32);
            if ( v29 != v28 + 48 )
              _libc_free(v29);
          }
          while ( v27 != v28 );
        }
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          v33 = v8 + 96 * v9;
          while ( v33 != v8 )
          {
            while ( 1 )
            {
              v33 -= 96LL;
              v34 = *(_QWORD *)(v33 + 32);
              if ( v34 == v33 + 48 )
                break;
              _libc_free(v34);
              if ( v33 == v8 )
                goto LABEL_44;
            }
          }
LABEL_44:
          *(_DWORD *)(a1 + 8) = 0;
          v9 = 0;
          sub_1995E60(a1, v11);
          v11 = *((unsigned int *)a2 + 2);
          v8 = *(_QWORD *)a1;
          v39 = *a2;
          v13 = *a2;
        }
        else
        {
          v13 = v39;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v30 = v8 + 32;
            v31 = (char **)(a2 + 6);
            v9 *= 96LL;
            v38 = v30 + v9;
            do
            {
              v32 = v30;
              v42 = v31;
              v30 += 96;
              *(_QWORD *)(v30 - 128) = *(v31 - 4);
              *(_QWORD *)(v30 - 120) = *(v31 - 3);
              *(_BYTE *)(v30 - 112) = *((_BYTE *)v31 - 16);
              *(_QWORD *)(v30 - 104) = *(v31 - 1);
              sub_19931B0(v32, v31, (__int64)v31, a4, v10, a6);
              v31 = v42 + 12;
              *(_QWORD *)(v30 - 48) = v42[6];
              *(_QWORD *)(v30 - 40) = v42[7];
            }
            while ( v30 != v38 );
            v11 = *((unsigned int *)a2 + 2);
            v8 = *(_QWORD *)a1;
            v39 = *a2;
            v13 = *a2 + v9;
          }
        }
        v14 = v8 + v9;
        v15 = v13;
        v16 = v39 + 96 * v11;
        if ( v16 != v13 )
        {
          do
          {
            if ( v14 )
            {
              *(_QWORD *)v14 = *(_QWORD *)v15;
              *(_QWORD *)(v14 + 8) = *(_QWORD *)(v15 + 8);
              *(_BYTE *)(v14 + 16) = *(_BYTE *)(v15 + 16);
              v17 = *(_QWORD *)(v15 + 24);
              *(_DWORD *)(v14 + 40) = 0;
              *(_QWORD *)(v14 + 24) = v17;
              *(_QWORD *)(v14 + 32) = v14 + 48;
              *(_DWORD *)(v14 + 44) = 4;
              v18 = *(unsigned int *)(v15 + 40);
              if ( (_DWORD)v18 )
              {
                v40 = v16;
                sub_19931B0(v14 + 32, (char **)(v15 + 32), v18, a4, v10, a6);
                v16 = v40;
              }
              *(_QWORD *)(v14 + 80) = *(_QWORD *)(v15 + 80);
              *(_QWORD *)(v14 + 88) = *(_QWORD *)(v15 + 88);
            }
            v15 += 96LL;
            v14 += 96LL;
          }
          while ( v16 != v15 );
        }
        *(_DWORD *)(a1 + 8) = v12;
        v19 = *a2;
        v20 = *a2 + 96LL * *((unsigned int *)a2 + 2);
        if ( *a2 != v20 )
        {
          do
          {
            v20 -= 96;
            v21 = *(_QWORD *)(v20 + 32);
            if ( v21 != v20 + 48 )
              _libc_free(v21);
          }
          while ( v19 != v20 );
        }
      }
      *((_DWORD *)a2 + 2) = 0;
    }
    else
    {
      v22 = v8 + 96 * v9;
      if ( v22 != v8 )
      {
        do
        {
          v22 -= 96LL;
          v23 = *(_QWORD *)(v22 + 32);
          if ( v23 != v22 + 48 )
            _libc_free(v23);
        }
        while ( v22 != v8 );
        v10 = *(_QWORD *)a1;
      }
      if ( v10 != a1 + 16 )
        _libc_free(v10);
      *(_QWORD *)a1 = *a2;
      *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
      *(_DWORD *)(a1 + 12) = *((_DWORD *)a2 + 3);
      a2[1] = 0;
      *a2 = v39;
    }
  }
}
