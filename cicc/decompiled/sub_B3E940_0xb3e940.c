// Function: sub_B3E940
// Address: 0xb3e940
//
void __fastcall sub_B3E940(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // r12
  unsigned __int64 i; // rbx
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // r15
  _QWORD *v13; // r12
  _QWORD *v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rbx
  __int64 v17; // rax
  _QWORD *v18; // r15
  _QWORD *v19; // r14
  _QWORD *v20; // r12
  _QWORD *v21; // rbx
  __int64 v22; // rsi
  int v23; // r12d
  __int64 v24; // r14
  __int64 *v25; // r15
  __int64 v26; // rdi
  __int64 v27; // r13
  __int64 v28; // rbx
  __int64 v29; // r12
  __int64 *v30; // r14
  __int64 v31; // rdi
  unsigned int v32; // [rsp+Ch] [rbp-64h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  unsigned __int64 v36; // [rsp+28h] [rbp-48h]
  unsigned __int64 v37; // [rsp+28h] [rbp-48h]
  unsigned __int64 v38; // [rsp+28h] [rbp-48h]
  __int64 v39[7]; // [rsp+38h] [rbp-38h] BYREF

  v33 = a2;
  if ( a1 != (__int64 *)a2 )
  {
    v2 = *((unsigned int *)a1 + 2);
    v3 = *a1;
    v32 = *(_DWORD *)(a2 + 8);
    v34 = v32;
    if ( v32 <= v2 )
    {
      v7 = *a1;
      if ( v32 )
      {
        v24 = v3 + 16;
        v25 = (__int64 *)(*(_QWORD *)a2 + 16LL);
        do
        {
          *(_DWORD *)(v24 - 16) = *((_DWORD *)v25 - 4);
          *(_DWORD *)(v24 - 12) = *((_DWORD *)v25 - 3);
          *(_BYTE *)(v24 - 8) = *((_BYTE *)v25 - 8);
          *(_BYTE *)(v24 - 7) = *((_BYTE *)v25 - 7);
          *(_BYTE *)(v24 - 6) = *((_BYTE *)v25 - 6);
          *(_BYTE *)(v24 - 5) = *((_BYTE *)v25 - 5);
          *(_DWORD *)(v24 - 4) = *((_DWORD *)v25 - 1);
          sub_B3C2C0(v24, v25);
          a2 = (__int64)(v25 + 6);
          v26 = v24 + 48;
          v24 += 192;
          sub_B3DB20(v26, (__int64)(v25 + 6));
          v25 += 24;
        }
        while ( v3 + 16 + 192LL * v32 != v24 );
        v3 += 192LL * v32;
        v7 = *a1;
        v2 = *((unsigned int *)a1 + 2);
      }
      v36 = v7 + 192 * v2;
      while ( v3 != v36 )
      {
        v36 -= 192LL;
        v8 = *(_QWORD *)(v36 + 64);
        v9 = v8 + 56LL * *(unsigned int *)(v36 + 72);
        if ( v8 != v9 )
        {
          do
          {
            v10 = *(unsigned int *)(v9 - 40);
            v11 = *(_QWORD **)(v9 - 48);
            v9 -= 56;
            v10 *= 32;
            v12 = (_QWORD *)((char *)v11 + v10);
            if ( v11 != (_QWORD *)((char *)v11 + v10) )
            {
              do
              {
                v12 -= 4;
                if ( (_QWORD *)*v12 != v12 + 2 )
                {
                  a2 = v12[2] + 1LL;
                  j_j___libc_free_0(*v12, a2);
                }
              }
              while ( v11 != v12 );
              v11 = *(_QWORD **)(v9 + 8);
            }
            if ( v11 != (_QWORD *)(v9 + 24) )
              _libc_free(v11, a2);
          }
          while ( v8 != v9 );
          v8 = *(_QWORD *)(v36 + 64);
        }
        if ( v8 != v36 + 80 )
          _libc_free(v8, a2);
        v13 = *(_QWORD **)(v36 + 16);
        v14 = &v13[4 * *(unsigned int *)(v36 + 24)];
        if ( v13 != v14 )
        {
          do
          {
            v14 -= 4;
            if ( (_QWORD *)*v14 != v14 + 2 )
            {
              a2 = v14[2] + 1LL;
              j_j___libc_free_0(*v14, a2);
            }
          }
          while ( v13 != v14 );
          v13 = *(_QWORD **)(v36 + 16);
        }
        if ( v13 != (_QWORD *)(v36 + 32) )
          _libc_free(v13, a2);
      }
    }
    else
    {
      if ( v32 > (unsigned __int64)*((unsigned int *)a1 + 3) )
      {
        v37 = v3 + 192 * v2;
        while ( v3 != v37 )
        {
          v37 -= 192LL;
          v15 = *(_QWORD *)(v37 + 64);
          v16 = v15 + 56LL * *(unsigned int *)(v37 + 72);
          if ( v15 != v16 )
          {
            do
            {
              v17 = *(unsigned int *)(v16 - 40);
              v18 = *(_QWORD **)(v16 - 48);
              v16 -= 56;
              v17 *= 32;
              v19 = (_QWORD *)((char *)v18 + v17);
              if ( v18 != (_QWORD *)((char *)v18 + v17) )
              {
                do
                {
                  v19 -= 4;
                  if ( (_QWORD *)*v19 != v19 + 2 )
                  {
                    a2 = v19[2] + 1LL;
                    j_j___libc_free_0(*v19, a2);
                  }
                }
                while ( v18 != v19 );
                v18 = *(_QWORD **)(v16 + 8);
              }
              if ( v18 != (_QWORD *)(v16 + 24) )
                _libc_free(v18, a2);
            }
            while ( v15 != v16 );
            v15 = *(_QWORD *)(v37 + 64);
          }
          if ( v15 != v37 + 80 )
            _libc_free(v15, a2);
          v20 = *(_QWORD **)(v37 + 16);
          v21 = &v20[4 * *(unsigned int *)(v37 + 24)];
          if ( v20 != v21 )
          {
            do
            {
              v21 -= 4;
              if ( (_QWORD *)*v21 != v21 + 2 )
              {
                a2 = v21[2] + 1LL;
                j_j___libc_free_0(*v21, a2);
              }
            }
            while ( v20 != v21 );
            v20 = *(_QWORD **)(v37 + 16);
          }
          if ( v20 != (_QWORD *)(v37 + 32) )
            _libc_free(v20, a2);
        }
        *((_DWORD *)a1 + 2) = 0;
        v22 = sub_C8D7D0(a1, a1 + 2, v32, 192, v39);
        v3 = v22;
        sub_B3DE10(a1, v22);
        v23 = v39[0];
        if ( a1 + 2 != (__int64 *)*a1 )
          _libc_free(*a1, v22);
        *a1 = v22;
        *((_DWORD *)a1 + 3) = v23;
        v34 = *(unsigned int *)(v33 + 8);
        v2 = 0;
      }
      else if ( *((_DWORD *)a1 + 2) )
      {
        v27 = v3 + 16;
        v28 = 192 * v2;
        v2 = v28;
        v29 = v27 + v28;
        v30 = (__int64 *)(*(_QWORD *)a2 + 16LL);
        do
        {
          v38 = v2;
          *(_DWORD *)(v27 - 16) = *((_DWORD *)v30 - 4);
          *(_DWORD *)(v27 - 12) = *((_DWORD *)v30 - 3);
          *(_BYTE *)(v27 - 8) = *((_BYTE *)v30 - 8);
          *(_BYTE *)(v27 - 7) = *((_BYTE *)v30 - 7);
          *(_BYTE *)(v27 - 6) = *((_BYTE *)v30 - 6);
          *(_BYTE *)(v27 - 5) = *((_BYTE *)v30 - 5);
          *(_DWORD *)(v27 - 4) = *((_DWORD *)v30 - 1);
          sub_B3C2C0(v27, v30);
          v31 = v27 + 48;
          v27 += 192;
          sub_B3DB20(v31, (__int64)(v30 + 6));
          v30 += 24;
          v2 = v38;
        }
        while ( v27 != v29 );
        v3 = *a1 + v28;
        v34 = *(unsigned int *)(a2 + 8);
      }
      v4 = *(_QWORD *)v33 + 192 * v34;
      for ( i = v2 + *(_QWORD *)v33; v4 != i; v3 += 192 )
      {
        if ( v3 )
        {
          *(_DWORD *)v3 = *(_DWORD *)i;
          *(_DWORD *)(v3 + 4) = *(_DWORD *)(i + 4);
          *(_BYTE *)(v3 + 8) = *(_BYTE *)(i + 8);
          *(_BYTE *)(v3 + 9) = *(_BYTE *)(i + 9);
          *(_BYTE *)(v3 + 10) = *(_BYTE *)(i + 10);
          *(_BYTE *)(v3 + 11) = *(_BYTE *)(i + 11);
          v6 = *(_DWORD *)(i + 12);
          *(_DWORD *)(v3 + 24) = 0;
          *(_DWORD *)(v3 + 12) = v6;
          *(_QWORD *)(v3 + 16) = v3 + 32;
          *(_DWORD *)(v3 + 28) = 1;
          if ( *(_DWORD *)(i + 24) )
            sub_B3C2C0(v3 + 16, (__int64 *)(i + 16));
          *(_DWORD *)(v3 + 72) = 0;
          *(_QWORD *)(v3 + 64) = v3 + 80;
          *(_DWORD *)(v3 + 76) = 2;
          if ( *(_DWORD *)(i + 72) )
            sub_B3DB20(v3 + 64, i + 64);
        }
        i += 192LL;
      }
    }
    *((_DWORD *)a1 + 2) = v32;
  }
}
