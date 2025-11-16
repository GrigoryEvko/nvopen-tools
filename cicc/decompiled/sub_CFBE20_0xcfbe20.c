// Function: sub_CFBE20
// Address: 0xcfbe20
//
__int64 __fastcall sub_CFBE20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _QWORD *v3; // rbx
  __int64 i; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 j; // rax
  __int64 v10; // rcx
  _QWORD *v11; // r8
  __int64 v12; // rdi
  _QWORD *v13; // r15
  __int64 v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // r14
  _QWORD *v17; // r15
  __int64 v18; // rax
  _QWORD *v19; // [rsp-110h] [rbp-110h]
  __int64 v20; // [rsp-108h] [rbp-108h]
  _QWORD *v21; // [rsp-100h] [rbp-100h]
  _QWORD v22[2]; // [rsp-F0h] [rbp-F0h] BYREF
  __int64 v23; // [rsp-E0h] [rbp-E0h]
  __int64 v24; // [rsp-D8h] [rbp-D8h]
  void *v25; // [rsp-C8h] [rbp-C8h]
  _QWORD v26[2]; // [rsp-C0h] [rbp-C0h] BYREF
  __int64 v27; // [rsp-B0h] [rbp-B0h]
  __int64 v28; // [rsp-A8h] [rbp-A8h]
  void *v29; // [rsp-98h] [rbp-98h]
  _QWORD v30[2]; // [rsp-90h] [rbp-90h] BYREF
  __int64 v31; // [rsp-80h] [rbp-80h]
  __int64 v32; // [rsp-78h] [rbp-78h]
  void *v33; // [rsp-68h] [rbp-68h]
  _QWORD v34[2]; // [rsp-60h] [rbp-60h] BYREF
  __int64 v35; // [rsp-50h] [rbp-50h]
  __int64 v36; // [rsp-48h] [rbp-48h]

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v3 = *(_QWORD **)(a1 + 8);
    v22[0] = 2;
    v22[1] = 0;
    v23 = -4096;
    v24 = 0;
    v26[0] = 2;
    v26[1] = 0;
    v27 = -8192;
    v25 = &unk_49DDB10;
    v28 = 0;
    v19 = &v3[6 * result];
    for ( i = -4096; ; i = v23 )
    {
      v5 = v3[3];
      if ( v5 != i )
      {
        i = v27;
        if ( v5 != v27 )
        {
          v6 = v3[5];
          i = v3[3];
          if ( v6 )
          {
            v7 = *(unsigned int *)(v6 + 184);
            if ( (_DWORD)v7 )
            {
              v8 = *(_QWORD *)(v6 + 168);
              v30[0] = 2;
              v30[1] = 0;
              v31 = -4096;
              v29 = &unk_49DDAE8;
              v33 = &unk_49DDAE8;
              v32 = 0;
              v34[0] = 2;
              v34[1] = 0;
              v35 = -8192;
              v36 = 0;
              v20 = v8 + 88 * v7;
              for ( j = -4096; ; j = v31 )
              {
                v10 = *(_QWORD *)(v8 + 24);
                if ( v10 != j )
                {
                  j = v35;
                  if ( v10 != v35 )
                  {
                    v11 = *(_QWORD **)(v8 + 40);
                    v12 = 4LL * *(unsigned int *)(v8 + 48);
                    v13 = &v11[v12];
                    if ( v11 != &v11[v12] )
                    {
                      do
                      {
                        v14 = *(v13 - 2);
                        v13 -= 4;
                        LOBYTE(a2) = v14 != -4096;
                        if ( ((v14 != 0) & (unsigned __int8)a2) != 0 && v14 != -8192 )
                        {
                          v21 = v11;
                          sub_BD60C0(v13);
                          v11 = v21;
                        }
                      }
                      while ( v11 != v13 );
                      v13 = *(_QWORD **)(v8 + 40);
                    }
                    if ( v13 != (_QWORD *)(v8 + 56) )
                      _libc_free(v13, a2);
                    j = *(_QWORD *)(v8 + 24);
                  }
                }
                *(_QWORD *)v8 = &unk_49DB368;
                LOBYTE(a2) = j != 0;
                if ( j != -4096 && j != 0 && j != -8192 )
                  sub_BD60C0((_QWORD *)(v8 + 8));
                v8 += 88;
                if ( v20 == v8 )
                  break;
              }
              v33 = &unk_49DB368;
              if ( v35 != -4096 && v35 != 0 && v35 != -8192 )
                sub_BD60C0(v34);
              v29 = &unk_49DB368;
              if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
                sub_BD60C0(v30);
              v7 = *(unsigned int *)(v6 + 184);
            }
            v15 = 88 * v7;
            sub_C7D6A0(*(_QWORD *)(v6 + 168), 88 * v7, 8);
            v16 = *(_QWORD **)(v6 + 16);
            v17 = &v16[4 * *(unsigned int *)(v6 + 24)];
            if ( v16 != v17 )
            {
              do
              {
                v18 = *(v17 - 2);
                v17 -= 4;
                if ( v18 != -4096 && v18 != 0 && v18 != -8192 )
                  sub_BD60C0(v17);
              }
              while ( v16 != v17 );
              v17 = *(_QWORD **)(v6 + 16);
            }
            if ( v17 != (_QWORD *)(v6 + 32) )
              _libc_free(v17, v15);
            a2 = 200;
            j_j___libc_free_0(v6, 200);
            i = v3[3];
          }
        }
      }
      *v3 = &unk_49DB368;
      if ( i != -4096 && i != 0 && i != -8192 )
        sub_BD60C0(v3 + 1);
      v3 += 6;
      if ( v19 == v3 )
        break;
    }
    v25 = &unk_49DB368;
    if ( v27 != -4096 && v27 != 0 && v27 != -8192 )
      sub_BD60C0(v26);
    result = v23;
    if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
      return sub_BD60C0(v22);
  }
  return result;
}
