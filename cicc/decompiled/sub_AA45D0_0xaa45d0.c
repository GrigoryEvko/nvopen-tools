// Function: sub_AA45D0
// Address: 0xaa45d0
//
void __fastcall sub_AA45D0(__int64 a1)
{
  _BYTE *v1; // rsi
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rdx
  _BYTE *v5; // r12
  __int64 v6; // rcx
  _BYTE *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  _BYTE *v12; // rcx
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned __int64 v18; // rdx
  _BYTE *v19; // rdx
  __int64 v20; // r12
  _QWORD *v21; // r15
  __int64 v22; // [rsp+18h] [rbp-78h]
  __int64 v23; // [rsp+18h] [rbp-78h]
  _QWORD *v24; // [rsp+18h] [rbp-78h]
  __int64 v25; // [rsp+18h] [rbp-78h]
  _BYTE *v26; // [rsp+28h] [rbp-68h] BYREF
  _BYTE *v27; // [rsp+30h] [rbp-60h] BYREF
  __int64 v28; // [rsp+38h] [rbp-58h]
  _BYTE v29[80]; // [rsp+40h] [rbp-50h] BYREF

  v1 = v29;
  v2 = a1 + 48;
  v28 = 0x400000000LL;
  v3 = *(_QWORD *)(a1 + 56);
  *(_BYTE *)(a1 + 40) = 1;
  v27 = v29;
  if ( v3 != a1 + 48 )
  {
    while ( 1 )
    {
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 8);
      v5 = (_BYTE *)(v4 - 24);
      if ( *(_BYTE *)(v4 - 24) != 85 )
        goto LABEL_3;
      v6 = *(_QWORD *)(v4 - 56);
      if ( !v6 )
        goto LABEL_3;
      if ( *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *(_QWORD *)(v4 + 56) || (*(_BYTE *)(v6 + 33) & 0x20) == 0 )
        goto LABEL_11;
      v13 = *(_DWORD *)(v6 + 36);
      if ( v13 > 0x45 )
      {
        if ( v13 == 71 )
          goto LABEL_28;
        goto LABEL_11;
      }
      if ( v13 > 0x43 )
      {
LABEL_28:
        v14 = sub_22077B0(96);
        v15 = v14;
        if ( v14 )
        {
          v1 = v5;
          sub_B13D50(v14, v5);
        }
        v16 = (unsigned int)v28;
        v17 = HIDWORD(v28);
        v18 = (unsigned int)v28 + 1LL;
        if ( v18 > HIDWORD(v28) )
        {
          v1 = v29;
          sub_C8D5F0(&v27, v29, v18, 8);
          v16 = (unsigned int)v28;
        }
        v19 = v27;
        *(_QWORD *)&v27[8 * v16] = v15;
        LODWORD(v28) = v28 + 1;
        sub_B43D60(v5, v1, v19, v17);
        if ( v2 == v3 )
          goto LABEL_5;
      }
      else
      {
LABEL_11:
        if ( !*(_BYTE *)v6
          && *(_QWORD *)(v6 + 24) == *(_QWORD *)(v4 + 56)
          && (*(_BYTE *)(v6 + 33) & 0x20) != 0
          && *(_DWORD *)(v6 + 36) == 70 )
        {
          v7 = *(_BYTE **)(v4 + 24);
          v8 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 - 20) & 0x7FFFFFF) - 24);
          v26 = v7;
          v9 = *(_QWORD *)(v8 + 24);
          if ( v7 )
            sub_B96E90(&v26, v7, 1);
          v10 = sub_22077B0(48);
          if ( v10 )
          {
            v22 = v10;
            sub_B12570(v10, v9, &v26);
            v10 = v22;
          }
          v11 = (unsigned int)v28;
          if ( (unsigned __int64)(unsigned int)v28 + 1 > HIDWORD(v28) )
          {
            v25 = v10;
            sub_C8D5F0(&v27, v29, (unsigned int)v28 + 1LL, 8);
            v11 = (unsigned int)v28;
            v10 = v25;
          }
          v12 = v27;
          *(_QWORD *)&v27[8 * v11] = v10;
          v1 = v26;
          LODWORD(v28) = v28 + 1;
          if ( v26 )
            sub_B91220(&v26);
          sub_B43D60(v5, v1, v11, v12);
          if ( v2 == v3 )
            goto LABEL_5;
        }
        else
        {
LABEL_3:
          if ( (_DWORD)v28 )
          {
            v23 = v4;
            sub_AA4580(a1, v4 - 24);
            v20 = *(_QWORD *)(v23 + 40);
            v21 = v27;
            v1 = &v27[8 * (unsigned int)v28];
            v24 = v1;
            if ( v1 != v27 )
            {
              do
              {
                v1 = (_BYTE *)*v21++;
                sub_B142B0(v20, v1, 0);
              }
              while ( v24 != v21 );
            }
            LODWORD(v28) = 0;
            if ( v2 == v3 )
            {
LABEL_5:
              if ( v27 != v29 )
                _libc_free(v27, v1);
              return;
            }
          }
          else if ( v2 == v3 )
          {
            goto LABEL_5;
          }
        }
      }
    }
  }
}
