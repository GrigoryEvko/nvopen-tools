// Function: sub_13EB2E0
// Address: 0x13eb2e0
//
void __fastcall sub_13EB2E0(__int64 *a1)
{
  __int64 v2; // rax
  __int64 v3; // r15
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  _QWORD *v6; // r12
  _QWORD *v7; // rbx
  unsigned __int64 v8; // rdi
  __int64 v9; // r14
  _QWORD *v10; // rbx
  _QWORD *v11; // r14
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // [rsp-48h] [rbp-48h]
  __int64 v20; // [rsp-48h] [rbp-48h]
  __int64 v21; // [rsp-40h] [rbp-40h]
  __int64 v22; // [rsp-40h] [rbp-40h]
  __int64 v23; // [rsp-40h] [rbp-40h]
  __int64 v24; // [rsp-40h] [rbp-40h]

  if ( a1[4] )
  {
    v2 = sub_13E7A30(a1 + 4, *a1, 0, 0);
    v3 = v2;
    if ( v2 )
    {
      j___libc_free_0(*(_QWORD *)(v2 + 248));
      v4 = *(_QWORD *)(v3 + 96);
      if ( v4 != v3 + 112 )
        _libc_free(v4);
      v5 = *(unsigned int *)(v3 + 88);
      if ( (_DWORD)v5 )
      {
        v6 = *(_QWORD **)(v3 + 72);
        v7 = &v6[10 * v5];
        do
        {
          if ( *v6 != -16 && *v6 != -8 )
          {
            v8 = v6[3];
            if ( v8 != v6[2] )
              _libc_free(v8);
          }
          v6 += 10;
        }
        while ( v7 != v6 );
      }
      j___libc_free_0(*(_QWORD *)(v3 + 72));
      v9 = *(unsigned int *)(v3 + 56);
      if ( (_DWORD)v9 )
      {
        v10 = *(_QWORD **)(v3 + 40);
        v11 = &v10[2 * v9];
        while ( 1 )
        {
          if ( *v10 != -16 && *v10 != -8 )
          {
            v12 = v10[1];
            if ( v12 )
              break;
          }
LABEL_29:
          v10 += 2;
          if ( v11 == v10 )
            goto LABEL_30;
        }
        if ( (*(_BYTE *)(v12 + 48) & 1) != 0 )
        {
          v14 = v12 + 56;
          v15 = v12 + 248;
        }
        else
        {
          v13 = *(unsigned int *)(v12 + 64);
          v14 = *(_QWORD *)(v12 + 56);
          if ( !(_DWORD)v13 )
            goto LABEL_34;
          v15 = v14 + 48 * v13;
        }
        do
        {
          if ( *(_QWORD *)v14 != -16 && *(_QWORD *)v14 != -8 && *(_DWORD *)(v14 + 8) == 3 )
          {
            if ( *(_DWORD *)(v14 + 40) > 0x40u )
            {
              v17 = *(_QWORD *)(v14 + 32);
              if ( v17 )
              {
                v19 = v15;
                v23 = v12;
                j_j___libc_free_0_0(v17);
                v15 = v19;
                v12 = v23;
              }
            }
            if ( *(_DWORD *)(v14 + 24) > 0x40u )
            {
              v18 = *(_QWORD *)(v14 + 16);
              if ( v18 )
              {
                v20 = v15;
                v24 = v12;
                j_j___libc_free_0_0(v18);
                v15 = v20;
                v12 = v24;
              }
            }
          }
          v14 += 48;
        }
        while ( v14 != v15 );
        if ( (*(_BYTE *)(v12 + 48) & 1) != 0 )
          goto LABEL_25;
        v14 = *(_QWORD *)(v12 + 56);
LABEL_34:
        v22 = v12;
        j___libc_free_0(v14);
        v12 = v22;
LABEL_25:
        *(_QWORD *)v12 = &unk_49EE2B0;
        v16 = *(_QWORD *)(v12 + 24);
        if ( v16 != 0 && v16 != -8 && v16 != -16 )
        {
          v21 = v12;
          sub_1649B30(v12 + 8);
          v12 = v21;
        }
        j_j___libc_free_0(v12, 248);
        goto LABEL_29;
      }
LABEL_30:
      j___libc_free_0(*(_QWORD *)(v3 + 40));
      j___libc_free_0(*(_QWORD *)(v3 + 8));
      j_j___libc_free_0(v3, 304);
    }
    a1[4] = 0;
  }
}
