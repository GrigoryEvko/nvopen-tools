// Function: sub_14EB1F0
// Address: 0x14eb1f0
//
__int64 __fastcall sub_14EB1F0(_QWORD *a1)
{
  __int64 result; // rax
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // rdi
  _QWORD *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // r12
  volatile signed __int32 *v10; // r14
  signed __int32 v11; // edx
  signed __int32 v12; // eax
  _QWORD *v13; // rbx
  _QWORD *v14; // r14
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // rdi
  _QWORD *v18; // rdi
  __int64 v19; // r15
  __int64 v20; // r12
  volatile signed __int32 *v21; // r13
  signed __int32 v22; // edx
  signed __int32 v23; // eax
  __int64 v24; // [rsp+8h] [rbp-68h]
  _QWORD *v25; // [rsp+10h] [rbp-60h]
  _QWORD *v26; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v27; // [rsp+28h] [rbp-48h]
  __int64 i; // [rsp+30h] [rbp-40h]
  char v29; // [rsp+38h] [rbp-38h]

  sub_15140E0(&v26, a1 + 3, 0);
  result = 1;
  if ( v29 )
  {
    v2 = (_QWORD *)a1[1];
    v24 = a1[2];
    v25 = (_QWORD *)*a1;
    *a1 = v26;
    v26 = 0;
    a1[1] = v27;
    v27 = 0;
    a1[2] = i;
    v3 = v25;
    for ( i = 0; v2 != v3; v3 += 11 )
    {
      v4 = v3[9];
      v5 = v3[8];
      if ( v4 != v5 )
      {
        do
        {
          v6 = *(_QWORD *)(v5 + 8);
          if ( v6 != v5 + 24 )
            j_j___libc_free_0(v6, *(_QWORD *)(v5 + 24) + 1LL);
          v5 += 40;
        }
        while ( v4 != v5 );
        v5 = v3[8];
      }
      if ( v5 )
        j_j___libc_free_0(v5, v3[10] - v5);
      v7 = (_QWORD *)v3[4];
      if ( v7 != v3 + 6 )
        j_j___libc_free_0(v7, v3[6] + 1LL);
      v8 = v3[2];
      v9 = v3[1];
      if ( v8 != v9 )
      {
        do
        {
          while ( 1 )
          {
            v10 = *(volatile signed __int32 **)(v9 + 8);
            if ( v10 )
            {
              if ( &_pthread_key_create )
              {
                v11 = _InterlockedExchangeAdd(v10 + 2, 0xFFFFFFFF);
              }
              else
              {
                v11 = *((_DWORD *)v10 + 2);
                *((_DWORD *)v10 + 2) = v11 - 1;
              }
              if ( v11 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 16LL))(v10);
                if ( &_pthread_key_create )
                {
                  v12 = _InterlockedExchangeAdd(v10 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v12 = *((_DWORD *)v10 + 3);
                  *((_DWORD *)v10 + 3) = v12 - 1;
                }
                if ( v12 == 1 )
                  break;
              }
            }
            v9 += 16;
            if ( v8 == v9 )
              goto LABEL_23;
          }
          v9 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 24LL))(v10);
        }
        while ( v8 != v9 );
LABEL_23:
        v9 = v3[1];
      }
      if ( v9 )
        j_j___libc_free_0(v9, v3[3] - v9);
    }
    if ( v25 )
      j_j___libc_free_0(v25, v24 - (_QWORD)v25);
    if ( v29 )
    {
      v13 = v27;
      v14 = v26;
      if ( v27 != v26 )
      {
        do
        {
          v15 = v14[9];
          v16 = v14[8];
          if ( v15 != v16 )
          {
            do
            {
              v17 = *(_QWORD *)(v16 + 8);
              if ( v17 != v16 + 24 )
                j_j___libc_free_0(v17, *(_QWORD *)(v16 + 24) + 1LL);
              v16 += 40;
            }
            while ( v15 != v16 );
            v16 = v14[8];
          }
          if ( v16 )
            j_j___libc_free_0(v16, v14[10] - v16);
          v18 = (_QWORD *)v14[4];
          if ( v18 != v14 + 6 )
            j_j___libc_free_0(v18, v14[6] + 1LL);
          v19 = v14[2];
          v20 = v14[1];
          if ( v19 != v20 )
          {
            do
            {
              while ( 1 )
              {
                v21 = *(volatile signed __int32 **)(v20 + 8);
                if ( v21 )
                {
                  if ( &_pthread_key_create )
                  {
                    v22 = _InterlockedExchangeAdd(v21 + 2, 0xFFFFFFFF);
                  }
                  else
                  {
                    v22 = *((_DWORD *)v21 + 2);
                    *((_DWORD *)v21 + 2) = v22 - 1;
                  }
                  if ( v22 == 1 )
                  {
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v21 + 16LL))(v21);
                    if ( &_pthread_key_create )
                    {
                      v23 = _InterlockedExchangeAdd(v21 + 3, 0xFFFFFFFF);
                    }
                    else
                    {
                      v23 = *((_DWORD *)v21 + 3);
                      *((_DWORD *)v21 + 3) = v23 - 1;
                    }
                    if ( v23 == 1 )
                      break;
                  }
                }
                v20 += 16;
                if ( v19 == v20 )
                  goto LABEL_51;
              }
              v20 += 16;
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v21 + 24LL))(v21);
            }
            while ( v19 != v20 );
LABEL_51:
            v20 = v14[1];
          }
          if ( v20 )
            j_j___libc_free_0(v20, v14[3] - v20);
          v14 += 11;
        }
        while ( v13 != v14 );
        v14 = v26;
      }
      if ( v14 )
        j_j___libc_free_0(v14, i - (_QWORD)v14);
    }
    return 0;
  }
  return result;
}
