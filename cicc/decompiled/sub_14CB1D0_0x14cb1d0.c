// Function: sub_14CB1D0
// Address: 0x14cb1d0
//
__int64 __fastcall sub_14CB1D0(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // rbx
  __int64 i; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 j; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rdi
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r14
  unsigned __int64 v15; // r15
  __int64 v16; // rax
  _QWORD *v17; // [rsp-110h] [rbp-110h]
  __int64 v18; // [rsp-108h] [rbp-108h]
  __int64 v19; // [rsp-100h] [rbp-100h]
  _QWORD v20[2]; // [rsp-F0h] [rbp-F0h] BYREF
  __int64 v21; // [rsp-E0h] [rbp-E0h]
  __int64 v22; // [rsp-D8h] [rbp-D8h]
  void *v23; // [rsp-C8h] [rbp-C8h]
  _QWORD v24[2]; // [rsp-C0h] [rbp-C0h] BYREF
  __int64 v25; // [rsp-B0h] [rbp-B0h]
  __int64 v26; // [rsp-A8h] [rbp-A8h]
  void *v27; // [rsp-98h] [rbp-98h]
  _QWORD v28[2]; // [rsp-90h] [rbp-90h] BYREF
  __int64 v29; // [rsp-80h] [rbp-80h]
  __int64 v30; // [rsp-78h] [rbp-78h]
  void *v31; // [rsp-68h] [rbp-68h]
  _QWORD v32[2]; // [rsp-60h] [rbp-60h] BYREF
  __int64 v33; // [rsp-50h] [rbp-50h]
  __int64 v34; // [rsp-48h] [rbp-48h]

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v2 = *(_QWORD **)(a1 + 8);
    v20[0] = 2;
    v20[1] = 0;
    v21 = -8;
    v22 = 0;
    v24[0] = 2;
    v24[1] = 0;
    v25 = -16;
    v23 = &unk_49ECBF8;
    v26 = 0;
    v17 = &v2[6 * result];
    for ( i = -8; ; i = v21 )
    {
      v4 = v2[3];
      if ( v4 != i )
      {
        i = v25;
        if ( v4 != v25 )
        {
          v5 = v2[5];
          i = v2[3];
          if ( v5 )
          {
            v6 = *(unsigned int *)(v5 + 176);
            if ( (_DWORD)v6 )
            {
              v7 = *(_QWORD *)(v5 + 160);
              v28[0] = 2;
              v28[1] = 0;
              v29 = -8;
              v27 = &unk_49ECBD0;
              v31 = &unk_49ECBD0;
              v30 = 0;
              v32[0] = 2;
              v32[1] = 0;
              v33 = -16;
              v34 = 0;
              v18 = v7 + 88 * v6;
              for ( j = -8; ; j = v29 )
              {
                v9 = *(_QWORD *)(v7 + 24);
                if ( v9 != j )
                {
                  j = v33;
                  if ( v9 != v33 )
                  {
                    v10 = *(_QWORD *)(v7 + 40);
                    v11 = 32LL * *(unsigned int *)(v7 + 48);
                    v12 = v10 + v11;
                    if ( v10 != v10 + v11 )
                    {
                      do
                      {
                        v13 = *(_QWORD *)(v12 - 16);
                        v12 -= 32LL;
                        if ( v13 != 0 && v13 != -8 && v13 != -16 )
                        {
                          v19 = v10;
                          sub_1649B30(v12);
                          v10 = v19;
                        }
                      }
                      while ( v10 != v12 );
                      v12 = *(_QWORD *)(v7 + 40);
                    }
                    if ( v12 != v7 + 56 )
                      _libc_free(v12);
                    j = *(_QWORD *)(v7 + 24);
                  }
                }
                *(_QWORD *)v7 = &unk_49EE2B0;
                if ( j != -8 && j != 0 && j != -16 )
                  sub_1649B30(v7 + 8);
                v7 += 88;
                if ( v18 == v7 )
                  break;
              }
              v31 = &unk_49EE2B0;
              if ( v33 != -8 && v33 != 0 && v33 != -16 )
                sub_1649B30(v32);
              v27 = &unk_49EE2B0;
              if ( v29 != 0 && v29 != -8 && v29 != -16 )
                sub_1649B30(v28);
            }
            j___libc_free_0(*(_QWORD *)(v5 + 160));
            v14 = *(_QWORD *)(v5 + 8);
            v15 = v14 + 32LL * *(unsigned int *)(v5 + 16);
            if ( v14 != v15 )
            {
              do
              {
                v16 = *(_QWORD *)(v15 - 16);
                v15 -= 32LL;
                if ( v16 != 0 && v16 != -8 && v16 != -16 )
                  sub_1649B30(v15);
              }
              while ( v14 != v15 );
              v15 = *(_QWORD *)(v5 + 8);
            }
            if ( v15 != v5 + 24 )
              _libc_free(v15);
            j_j___libc_free_0(v5, 192);
            i = v2[3];
          }
        }
      }
      *v2 = &unk_49EE2B0;
      if ( i != -8 && i != 0 && i != -16 )
        sub_1649B30(v2 + 1);
      v2 += 6;
      if ( v17 == v2 )
        break;
    }
    v23 = &unk_49EE2B0;
    if ( v25 != 0 && v25 != -8 && v25 != -16 )
      sub_1649B30(v24);
    result = v21;
    if ( v21 != 0 && v21 != -8 && v21 != -16 )
      return sub_1649B30(v20);
  }
  return result;
}
