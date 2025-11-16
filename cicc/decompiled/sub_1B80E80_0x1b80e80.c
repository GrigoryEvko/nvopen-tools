// Function: sub_1B80E80
// Address: 0x1b80e80
//
void __fastcall sub_1B80E80(__int64 a1)
{
  __int64 v1; // rcx
  unsigned int v2; // edx
  _QWORD *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int8 v10; // al
  int v11; // r8d
  int v12; // r9d
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 *v15; // rsi
  __int64 *v16; // rcx
  _QWORD *v17; // r14
  _QWORD *v18; // r12
  __int64 v19; // r8
  _QWORD *v20; // r15
  _QWORD *v21; // [rsp+20h] [rbp-3A0h] BYREF
  __int64 v22; // [rsp+28h] [rbp-398h]
  _QWORD v23[16]; // [rsp+30h] [rbp-390h] BYREF
  __int64 v24; // [rsp+B0h] [rbp-310h] BYREF
  __int64 *v25; // [rsp+B8h] [rbp-308h]
  __int64 *v26; // [rsp+C0h] [rbp-300h]
  __int64 v27; // [rsp+C8h] [rbp-2F8h]
  int v28; // [rsp+D0h] [rbp-2F0h]
  _BYTE v29[136]; // [rsp+D8h] [rbp-2E8h] BYREF
  _BYTE v30[16]; // [rsp+160h] [rbp-260h] BYREF
  __int64 v31; // [rsp+170h] [rbp-250h]

  sub_143ACA0((__int64)v30, *(_QWORD *)(a1 + 40));
  v1 = *(_QWORD *)(a1 + 40);
  v25 = (__int64 *)v29;
  v2 = 1;
  v26 = (__int64 *)v29;
  v3 = v23;
  v22 = 0x1000000001LL;
  v24 = 0;
  v27 = 16;
  v28 = 0;
  v21 = v23;
  v23[0] = a1;
  while ( 2 )
  {
    v4 = v2--;
    v5 = v3[v4 - 1];
    LODWORD(v22) = v2;
    if ( (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) == 0 )
      goto LABEL_14;
    v6 = 0;
    v7 = 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
    do
    {
      while ( 1 )
      {
        v8 = (*(_BYTE *)(v5 + 23) & 0x40) != 0 ? *(_QWORD *)(v5 - 8) : v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
        v9 = *(_QWORD *)(v8 + v6);
        v10 = *(_BYTE *)(v9 + 16);
        if ( v10 > 0x17u && v10 != 77 && *(_QWORD *)(v9 + 40) == v1 )
          break;
        v6 += 24;
        if ( v6 == v7 )
          goto LABEL_13;
      }
      if ( !sub_143B490((__int64)v30, v9, a1) )
      {
        v13 = v25;
        if ( v26 != v25 )
          goto LABEL_17;
        v15 = &v25[HIDWORD(v27)];
        if ( v25 != v15 )
        {
          v16 = 0;
          while ( v9 != *v13 )
          {
            if ( *v13 == -2 )
              v16 = v13;
            if ( v15 == ++v13 )
            {
              if ( !v16 )
                goto LABEL_42;
              *v16 = v9;
              v14 = (unsigned int)v22;
              --v28;
              ++v24;
              if ( (unsigned int)v22 < HIDWORD(v22) )
                goto LABEL_19;
              goto LABEL_28;
            }
          }
          goto LABEL_18;
        }
LABEL_42:
        if ( HIDWORD(v27) < (unsigned int)v27 )
        {
          ++HIDWORD(v27);
          *v15 = v9;
          ++v24;
        }
        else
        {
LABEL_17:
          sub_16CCBA0((__int64)&v24, v9);
        }
LABEL_18:
        v14 = (unsigned int)v22;
        if ( (unsigned int)v22 >= HIDWORD(v22) )
        {
LABEL_28:
          sub_16CD150((__int64)&v21, v23, 0, 8, v11, v12);
          v14 = (unsigned int)v22;
        }
LABEL_19:
        v21[v14] = v9;
        LODWORD(v22) = v22 + 1;
      }
      v6 += 24;
      v1 = *(_QWORD *)(a1 + 40);
    }
    while ( v6 != v7 );
LABEL_13:
    v2 = v22;
LABEL_14:
    if ( v2 )
    {
      v3 = v21;
      continue;
    }
    break;
  }
  v17 = (_QWORD *)(a1 + 24);
  v18 = (_QWORD *)(v1 + 40);
  if ( a1 + 24 != v1 + 40 )
  {
    do
    {
      v19 = (__int64)(v17 - 3);
      if ( !v17 )
        v19 = 0;
      v20 = (_QWORD *)v19;
      if ( sub_13A0E30((__int64)&v24, v19) )
      {
        v17 = (_QWORD *)(*v17 & 0xFFFFFFFFFFFFFFF8LL);
        sub_15F2070(v20);
        sub_15F2120((__int64)v20, a1);
      }
      v17 = (_QWORD *)v17[1];
    }
    while ( v18 != v17 );
  }
  if ( v21 != v23 )
    _libc_free((unsigned __int64)v21);
  if ( v26 != v25 )
    _libc_free((unsigned __int64)v26);
  if ( (v30[8] & 1) == 0 )
    j___libc_free_0(v31);
}
