// Function: sub_35E5750
// Address: 0x35e5750
//
void __fastcall sub_35E5750(__int64 a1)
{
  __int64 v1; // r10
  __int64 v2; // rsi
  __int64 v3; // r13
  unsigned __int64 v4; // r14
  __int64 v5; // r12
  __int64 v6; // r8
  int v7; // ebx
  _DWORD *v8; // rax
  __int64 v9; // rdx
  _DWORD *v10; // rsi
  __int64 v11; // r9
  __int64 v12; // rdx
  _DWORD *v13; // rdx
  _BYTE *v14; // r8
  __int64 v15; // r9
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  __int64 v18; // [rsp+8h] [rbp-258h]
  __int64 v19; // [rsp+10h] [rbp-250h]
  _BYTE *v20; // [rsp+20h] [rbp-240h] BYREF
  __int64 v21; // [rsp+28h] [rbp-238h]
  _BYTE v22[560]; // [rsp+30h] [rbp-230h] BYREF

  v1 = a1;
  v2 = *(_QWORD *)(a1 + 24);
  v20 = v22;
  v21 = 0x8000000000LL;
  v3 = *(_QWORD *)(v2 + 56);
  v4 = v2 + 48;
  if ( v3 == v2 + 48 )
  {
    v16 = v2 + 48;
    v15 = 0;
    v14 = v22;
  }
  else
  {
    do
    {
      v5 = *(_QWORD *)(v3 + 32);
      v6 = v5 + 40LL * (*(_DWORD *)(v3 + 40) & 0xFFFFFF);
      if ( v5 != v6 )
      {
        while ( 1 )
        {
          if ( *(_BYTE *)v5 )
            goto LABEL_13;
          v7 = *(_DWORD *)(v5 + 8);
          if ( !v7 )
            goto LABEL_13;
          v8 = v20;
          v9 = 4LL * (unsigned int)v21;
          v10 = &v20[v9];
          v11 = v9 >> 2;
          v12 = v9 >> 4;
          if ( v12 )
          {
            v13 = &v20[16 * v12];
            while ( v7 != *v8 )
            {
              if ( v7 == v8[1] )
              {
                ++v8;
                goto LABEL_12;
              }
              if ( v7 == v8[2] )
              {
                v8 += 2;
                goto LABEL_12;
              }
              if ( v7 == v8[3] )
              {
                v8 += 3;
                goto LABEL_12;
              }
              v8 += 4;
              if ( v13 == v8 )
              {
                v11 = v10 - v8;
                goto LABEL_24;
              }
            }
            goto LABEL_12;
          }
LABEL_24:
          if ( v11 == 2 )
            goto LABEL_31;
          if ( v11 == 3 )
            break;
          if ( v11 != 1 )
          {
LABEL_27:
            v17 = (unsigned int)v21 + 1LL;
            if ( v17 > HIDWORD(v21) )
              goto LABEL_35;
            goto LABEL_28;
          }
LABEL_33:
          if ( v7 != *v8 )
          {
            v17 = (unsigned int)v21 + 1LL;
            if ( v17 > HIDWORD(v21) )
            {
LABEL_35:
              v18 = v1;
              v19 = v6;
              sub_C8D5F0((__int64)&v20, v22, v17, 4u, v6, v11);
              v1 = v18;
              v6 = v19;
              v10 = &v20[4 * (unsigned int)v21];
            }
LABEL_28:
            *v10 = v7;
            LODWORD(v21) = v21 + 1;
            goto LABEL_13;
          }
LABEL_12:
          if ( v10 == v8 )
            goto LABEL_27;
LABEL_13:
          v5 += 40;
          if ( v6 == v5 )
            goto LABEL_14;
        }
        if ( v7 == *v8 )
          goto LABEL_12;
        ++v8;
LABEL_31:
        if ( v7 == *v8 )
          goto LABEL_12;
        ++v8;
        goto LABEL_33;
      }
LABEL_14:
      if ( (*(_BYTE *)v3 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v3 + 44) & 8) != 0 )
          v3 = *(_QWORD *)(v3 + 8);
      }
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v4 != v3 );
    v2 = *(_QWORD *)(v1 + 24);
    v14 = v20;
    v15 = (unsigned int)v21;
    v4 = *(_QWORD *)(v2 + 56);
    v16 = v2 + 48;
  }
  sub_2E17AE0(*(_QWORD *)(*(_QWORD *)(v1 + 8) + 48LL), v2, v4, v16, v14, v15);
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
}
