// Function: sub_186F120
// Address: 0x186f120
//
__int64 __fastcall sub_186F120(_QWORD *a1, _QWORD *a2, int a3)
{
  _QWORD *v3; // rbx
  __int64 v5; // r12
  unsigned __int64 v6; // r8
  __int64 v7; // r13
  __int64 v8; // r13
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  __int64 *v11; // r13
  _QWORD *v12; // rax
  _QWORD *v13; // r14
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 *v21; // r14
  size_t v22; // rbx
  __int64 v23; // r8
  _BYTE *v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // rax
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  __int64 *v30; // [rsp+18h] [rbp-58h]
  _QWORD *v31; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+28h] [rbp-48h]
  unsigned __int64 v33; // [rsp+30h] [rbp-40h]
  __int64 v34; // [rsp+30h] [rbp-40h]
  unsigned __int64 v35; // [rsp+38h] [rbp-38h]
  __int64 v36; // [rsp+38h] [rbp-38h]

  v3 = a1;
  switch ( a3 )
  {
    case 1:
      *a1 = *a2;
      return 0;
    case 2:
      v11 = (__int64 *)*a2;
      v12 = (_QWORD *)sub_22077B0(32);
      v13 = v12;
      if ( !v12 )
        goto LABEL_29;
      *v12 = 0;
      v12[1] = 0;
      v12[2] = 0x1000000000LL;
      if ( !*((_DWORD *)v11 + 3)
        || (sub_16D1890((__int64)v12, *((_DWORD *)v11 + 2)),
            v14 = *v13,
            v15 = *v11,
            v16 = *((unsigned int *)v13 + 2),
            v29 = *v13,
            v28 = *v11,
            *(_QWORD *)((char *)v13 + 12) = *(__int64 *)((char *)v11 + 12),
            !(_DWORD)v16) )
      {
LABEL_29:
        *v3 = v13;
        return 0;
      }
      v30 = v11;
      v17 = 0;
      v31 = v13;
      v18 = 8 * v16 + 8;
      v32 = 8LL * (unsigned int)(v16 - 1);
      v19 = v15;
      while ( 1 )
      {
        v20 = *(_QWORD *)(v19 + v17);
        v21 = (__int64 *)(v14 + v17);
        if ( v20 )
        {
          if ( v20 != -8 )
            break;
        }
        *v21 = v20;
LABEL_20:
        v18 += 4;
        if ( v32 == v17 )
        {
          v13 = v31;
          v3 = a1;
          goto LABEL_29;
        }
        v17 += 8;
        v19 = *v30;
        v14 = *v31;
      }
      v22 = *(_QWORD *)v20;
      v33 = *(_QWORD *)v20 + 17LL;
      v35 = *(_QWORD *)v20 + 1LL;
      v23 = malloc(v33);
      if ( !v23 )
      {
        if ( !v33 )
        {
          v26 = malloc(1u);
          v23 = 0;
          if ( v26 )
          {
            v24 = (_BYTE *)(v26 + 16);
            v23 = v26;
            goto LABEL_26;
          }
        }
        v34 = v23;
        sub_16BD1C0("Allocation failed", 1u);
        v23 = v34;
      }
      v24 = (_BYTE *)(v23 + 16);
      if ( v35 <= 1 )
      {
LABEL_19:
        v24[v22] = 0;
        *(_QWORD *)v23 = v22;
        *(_BYTE *)(v23 + 8) = *(_BYTE *)(v20 + 8);
        *v21 = v23;
        *(_DWORD *)(v29 + v18) = *(_DWORD *)(v28 + v18);
        goto LABEL_20;
      }
LABEL_26:
      v36 = v23;
      v25 = memcpy(v24, (const void *)(v20 + 16), v22);
      v23 = v36;
      v24 = v25;
      goto LABEL_19;
    case 3:
      v5 = *a1;
      if ( *a1 )
      {
        v6 = *(_QWORD *)v5;
        if ( *(_DWORD *)(v5 + 12) )
        {
          v7 = *(unsigned int *)(v5 + 8);
          if ( (_DWORD)v7 )
          {
            v8 = 8 * v7;
            v9 = 0;
            do
            {
              v10 = *(_QWORD *)(v6 + v9);
              if ( v10 != -8 && v10 )
              {
                _libc_free(v10);
                v6 = *(_QWORD *)v5;
              }
              v9 += 8;
            }
            while ( v8 != v9 );
          }
        }
        _libc_free(v6);
        j_j___libc_free_0(v5, 32);
      }
      break;
  }
  return 0;
}
