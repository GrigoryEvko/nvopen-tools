// Function: sub_23993D0
// Address: 0x23993d0
//
__int64 __fastcall sub_23993D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned int v3; // eax
  int v4; // eax
  __int64 v5; // rax
  unsigned int v6; // eax
  __int64 *v7; // r12
  _QWORD *v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 result; // rax
  __int64 v13; // r15
  _QWORD *v14; // r13
  _QWORD *v15; // rbx
  bool v16; // dl
  __int64 v17; // rsi
  _QWORD *v18; // r12
  __int64 v19; // rax
  int v20; // edx
  _QWORD *v21; // rax
  __int64 *v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r9
  _QWORD *v26; // rax
  __int64 v27; // [rsp+8h] [rbp-A8h]
  unsigned int v28; // [rsp+18h] [rbp-98h]
  _QWORD *v29; // [rsp+18h] [rbp-98h]
  __int64 v30; // [rsp+20h] [rbp-90h]
  __int64 v31; // [rsp+28h] [rbp-88h] BYREF
  __int64 v32; // [rsp+30h] [rbp-80h]
  __int64 v33; // [rsp+38h] [rbp-78h] BYREF
  unsigned int v34; // [rsp+40h] [rbp-70h]
  char v35; // [rsp+78h] [rbp-38h] BYREF

  v2 = a1;
  v3 = *(_DWORD *)(a2 + 8) & 0xFFFFFFFE;
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a1 + 8) & 0xFFFFFFFE | *(_DWORD *)(a2 + 8) & 1;
  *(_DWORD *)(a1 + 8) = v3 | *(_DWORD *)(a1 + 8) & 1;
  v4 = *(_DWORD *)(a1 + 12);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  *(_DWORD *)(a2 + 12) = v4;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
      goto LABEL_4;
    v13 = a1 + 24;
    v14 = (_QWORD *)(a2 + 104);
    v29 = (_QWORD *)(a2 + 456);
    while ( 1 )
    {
      result = *(_QWORD *)(v13 - 8);
      v15 = v14 - 11;
      v16 = result != -4096 && result != -8192;
      v17 = *(v14 - 11);
      if ( v17 == -4096 )
      {
        *(_QWORD *)(v13 - 8) = -4096;
        *(v14 - 11) = result;
        if ( v16 )
          goto LABEL_30;
      }
      else if ( v17 == -8192 )
      {
        *(_QWORD *)(v13 - 8) = -8192;
        *(v14 - 11) = result;
        if ( v16 )
        {
LABEL_30:
          *(v14 - 10) = 0;
          v21 = v14 - 8;
          *((_DWORD *)v15 + 4) = 1;
          *((_DWORD *)v14 - 17) = 0;
          do
          {
            if ( v21 )
              *v21 = -4096;
            v21 += 2;
          }
          while ( v14 != v21 );
          result = sub_1033120((__int64)(v14 - 10), v13);
          if ( (*(_BYTE *)(v13 - 8 + 16) & 1) == 0 )
            result = sub_C7D6A0(*(_QWORD *)(v13 + 16), 16LL * *(unsigned int *)(v13 + 24), 8);
        }
      }
      else
      {
        v18 = (_QWORD *)(v13 + 16);
        if ( v16 )
        {
          v31 = 0;
          v32 = 1;
          v30 = result;
          v22 = &v33;
          do
          {
            *v22 = -4096;
            v22 += 2;
          }
          while ( v22 != (__int64 *)&v35 );
          sub_1033120((__int64)&v31, v13);
          v23 = v13 - 8;
          v24 = (__int64)(v14 - 10);
          *(_QWORD *)(v13 - 8) = *(v14 - 11);
          if ( (*(_BYTE *)(v13 - 8 + 16) & 1) == 0 )
          {
            sub_C7D6A0(*(_QWORD *)(v13 + 16), 16LL * *(unsigned int *)(v13 + 24), 8);
            v24 = (__int64)(v14 - 10);
            v23 = v13 - 8;
          }
          *(_DWORD *)(v23 + 16) = 1;
          *(_DWORD *)(v13 + 12) = 0;
          do
          {
            if ( v18 )
              *v18 = -4096;
            v18 += 2;
          }
          while ( v18 != (_QWORD *)(v13 + 80) );
          v27 = v24;
          sub_1033120(v13, v24);
          v25 = v27;
          *(v14 - 11) = v30;
          if ( (v15[2] & 1) == 0 )
          {
            sub_C7D6A0(*(v14 - 8), 16LL * *((unsigned int *)v14 - 14), 8);
            v25 = v27;
          }
          *((_DWORD *)v15 + 4) = 1;
          v26 = v14 - 8;
          *((_DWORD *)v14 - 17) = 0;
          do
          {
            if ( v26 )
              *v26 = -4096;
            v26 += 2;
          }
          while ( v14 != v26 );
          result = sub_1033120(v25, (__int64)&v31);
          if ( (v32 & 1) == 0 )
            result = sub_C7D6A0(v33, 16LL * v34, 8);
        }
        else
        {
          *(_QWORD *)(v13 - 8) = v17;
          *(v14 - 11) = result;
          *(_QWORD *)v13 = 0;
          *(_DWORD *)(v13 - 8 + 16) = 1;
          *(_DWORD *)(v13 + 12) = 0;
          do
          {
            if ( v18 )
              *v18 = -4096;
            v18 += 2;
          }
          while ( v18 != (_QWORD *)(v13 + 80) );
          result = sub_1033120(v13, (__int64)(v14 - 10));
          if ( (v15[2] & 1) == 0 )
            result = sub_C7D6A0(*(v14 - 8), 16LL * *((unsigned int *)v14 - 14), 8);
        }
      }
      v13 += 88;
      v14 += 11;
      if ( v14 == v29 )
        return result;
    }
  }
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
  {
    v19 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    v20 = *(_DWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = v19;
    result = *(unsigned int *)(a1 + 24);
    *(_DWORD *)(a1 + 24) = v20;
    *(_DWORD *)(a2 + 24) = result;
    return result;
  }
  v5 = a2;
  a2 = a1;
  v2 = v5;
LABEL_4:
  v6 = *(_DWORD *)(a2 + 24);
  *(_BYTE *)(a2 + 8) |= 1u;
  v7 = (__int64 *)(v2 + 16);
  v8 = (_QWORD *)(a2 + 104);
  v9 = *(_QWORD *)(a2 + 16);
  v28 = v6;
  do
  {
    v10 = *v7;
    *(v8 - 11) = *v7;
    if ( v10 != -4096 && v10 != -8192 )
    {
      *(v8 - 10) = 0;
      *((_DWORD *)v8 - 18) = 1;
      v11 = v8 - 8;
      *((_DWORD *)v8 - 17) = 0;
      do
      {
        if ( v11 )
          *v11 = -4096;
        v11 += 2;
      }
      while ( v11 != v8 );
      sub_1033120((__int64)(v8 - 10), (__int64)(v7 + 1));
      if ( (v7[2] & 1) == 0 )
        sub_C7D6A0(v7[3], 16LL * *((unsigned int *)v7 + 8), 8);
    }
    v7 += 11;
    v8 += 11;
  }
  while ( v7 != (__int64 *)(v2 + 368) );
  *(_BYTE *)(v2 + 8) &= ~1u;
  *(_QWORD *)(v2 + 16) = v9;
  *(_DWORD *)(v2 + 24) = v28;
  return v28;
}
