// Function: sub_1B2F000
// Address: 0x1b2f000
//
void __fastcall sub_1B2F000(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rcx
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // r15
  __int64 v16; // r12
  __int64 *v17; // rbx
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // [rsp+20h] [rbp-F0h]
  __int64 *v22; // [rsp+28h] [rbp-E8h]
  __int64 *v23; // [rsp+30h] [rbp-E0h]
  _BYTE v24[4]; // [rsp+4Ch] [rbp-C4h] BYREF
  __int64 *v25; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v26; // [rsp+58h] [rbp-B8h]
  _BYTE v27[16]; // [rsp+60h] [rbp-B0h] BYREF
  _DWORD *v28[4]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v29; // [rsp+90h] [rbp-80h] BYREF
  __int64 v30; // [rsp+98h] [rbp-78h]
  _BYTE v31[112]; // [rsp+A0h] [rbp-70h] BYREF

  v29 = (__int64 *)v31;
  v30 = 0x800000000LL;
  v25 = (__int64 *)v27;
  v26 = 0x200000000LL;
  v5 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v28[0] = v24;
  v28[2] = v24;
  if ( (unsigned __int8)sub_1B2BDF0(v28, v5) )
  {
    v9 = *(_QWORD *)(v5 - 48);
    v10 = (unsigned int)v26;
    if ( (unsigned int)v26 >= HIDWORD(v26) )
    {
      sub_16CD150((__int64)&v25, v27, 0, 8, v7, v8);
      v10 = (unsigned int)v26;
    }
    v25[v10] = v9;
    v11 = (unsigned int)(v26 + 1);
    LODWORD(v26) = v11;
    v12 = *(_QWORD *)(v5 - 24);
    if ( (unsigned int)v11 >= HIDWORD(v26) )
    {
      sub_16CD150((__int64)&v25, v27, 0, 8, v7, v8);
      v11 = (unsigned int)v26;
    }
    v25[v11] = v12;
    v13 = (unsigned int)(v26 + 1);
    LODWORD(v26) = v13;
    if ( HIDWORD(v26) > (unsigned int)v13 )
      goto LABEL_7;
    goto LABEL_30;
  }
  v13 = (unsigned int)v26;
  if ( (unsigned __int8)(*(_BYTE *)(v5 + 16) - 75) <= 1u )
  {
    if ( HIDWORD(v26) > (unsigned int)v26 )
    {
LABEL_7:
      v25[v13] = v5;
      v13 = (unsigned int)(v26 + 1);
      LODWORD(v26) = v26 + 1;
      goto LABEL_8;
    }
LABEL_30:
    sub_16CD150((__int64)&v25, v27, 0, 8, v7, v8);
    v13 = (unsigned int)v26;
    goto LABEL_7;
  }
LABEL_8:
  v14 = (__int64)v25;
  v22 = &v25[v13];
  if ( v25 != v22 )
  {
    v15 = v25;
    do
    {
      v16 = *v15;
      if ( (unsigned __int8)(*(_BYTE *)(*v15 + 16) - 75) > 1u )
      {
        v20 = sub_22077B0(56);
        if ( v20 )
        {
          *(_QWORD *)(v20 + 8) = 0;
          *(_QWORD *)(v20 + 16) = 0;
          *(_DWORD *)(v20 + 24) = 1;
          *(_QWORD *)(v20 + 32) = v16;
          *(_QWORD *)(v20 + 40) = v16;
          *(_QWORD *)v20 = &unk_49F6720;
          *(_QWORD *)(v20 + 48) = a2;
        }
        sub_1B2EEE0(a1, a4, v16, v20);
      }
      else
      {
        sub_1B2B720(*v15, (__int64)&v29, v14, v6, v7, v8);
        v17 = v29;
        v23 = &v29[(unsigned int)v30];
        if ( v29 != v23 )
        {
          v21 = v15;
          do
          {
            v18 = *v17;
            v19 = sub_22077B0(56);
            if ( v19 )
            {
              *(_QWORD *)(v19 + 8) = 0;
              *(_QWORD *)(v19 + 16) = 0;
              *(_DWORD *)(v19 + 24) = 1;
              *(_QWORD *)(v19 + 32) = v18;
              *(_QWORD *)(v19 + 40) = v16;
              *(_QWORD *)v19 = &unk_49F6720;
              *(_QWORD *)(v19 + 48) = a2;
            }
            ++v17;
            sub_1B2EEE0(a1, a4, v18, v19);
          }
          while ( v23 != v17 );
          v15 = v21;
        }
        LODWORD(v30) = 0;
      }
      ++v15;
    }
    while ( v22 != v15 );
    v22 = v25;
  }
  if ( v22 != (__int64 *)v27 )
    _libc_free((unsigned __int64)v22);
  if ( v29 != (__int64 *)v31 )
    _libc_free((unsigned __int64)v29);
}
