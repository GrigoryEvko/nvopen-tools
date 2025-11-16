// Function: sub_29F8B90
// Address: 0x29f8b90
//
__int64 __fastcall sub_29F8B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 *v11; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  int v15; // r12d
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 i; // r13
  _BYTE *v19; // rsi
  int *v20; // r15
  int *v21; // r12
  int *v22; // rsi
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+18h] [rbp-68h]
  int v25; // [rsp+20h] [rbp-60h] BYREF
  __int64 v26; // [rsp+28h] [rbp-58h]
  unsigned int v27; // [rsp+38h] [rbp-48h]
  int *v28; // [rsp+40h] [rbp-40h]
  int v29; // [rsp+48h] [rbp-38h]
  char v30; // [rsp+50h] [rbp-30h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  if ( !*(_BYTE *)(a3 + 28) )
  {
    if ( sub_C8CA60(a3, a2) )
      return a1;
    if ( !*(_BYTE *)(a3 + 28) )
      goto LABEL_23;
    v8 = *(__int64 **)(a3 + 8);
    v10 = *(unsigned int *)(a3 + 20);
    v11 = &v8[v10];
    if ( v8 != v11 )
    {
LABEL_8:
      while ( a2 != *v8 )
      {
        if ( ++v8 == v11 )
          goto LABEL_36;
      }
LABEL_9:
      if ( (unsigned __int8)sub_B46970((unsigned __int8 *)a2) )
        goto LABEL_10;
      goto LABEL_24;
    }
LABEL_36:
    if ( *(_DWORD *)(a3 + 16) > (unsigned int)v10 )
    {
      *(_DWORD *)(a3 + 20) = v10 + 1;
      *v11 = a2;
      ++*(_QWORD *)a3;
      goto LABEL_9;
    }
LABEL_23:
    sub_C8CC70(a3, a2, (__int64)v8, v9, a5, a6);
    if ( (unsigned __int8)sub_B46970((unsigned __int8 *)a2) )
    {
LABEL_10:
      v13 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL);
      v14 = *(_QWORD *)(v13 + 80);
      v23 = v13 + 72;
      if ( v14 != v13 + 72 )
      {
        v15 = 0;
        do
        {
          if ( !v14 )
            BUG();
          v16 = *(_QWORD *)(v14 + 32);
          v17 = v14 + 24;
          if ( v16 != v14 + 24 )
          {
            do
            {
              while ( !v16 || a2 != v16 - 24 )
              {
                v16 = *(_QWORD *)(v16 + 8);
                ++v15;
                if ( v17 == v16 )
                  goto LABEL_19;
              }
              v24 = v17;
              v25 = v15++;
              sub_29F8930(a1, &v25);
              v17 = v24;
              v16 = *(_QWORD *)(v16 + 8);
            }
            while ( v24 != v16 );
          }
LABEL_19:
          v14 = *(_QWORD *)(v14 + 8);
        }
        while ( v23 != v14 );
      }
      return a1;
    }
LABEL_24:
    if ( *(_BYTE *)a2 != 30 )
    {
      for ( i = *(_QWORD *)(a2 + 16); i; i = *(_QWORD *)(i + 8) )
      {
        v19 = *(_BYTE **)(i + 24);
        if ( *v19 > 0x1Cu )
        {
          sub_29F8B90(&v25, v19, a3);
          v20 = v28;
          v21 = &v28[v29];
          if ( v28 != v21 )
          {
            do
            {
              v22 = v20++;
              sub_29F8930(a1, v22);
            }
            while ( v21 != v20 );
            v21 = v28;
          }
          if ( v21 != (int *)&v30 )
            _libc_free((unsigned __int64)v21);
          sub_C7D6A0(v26, 4LL * v27, 4);
        }
      }
      return a1;
    }
    goto LABEL_10;
  }
  v8 = *(__int64 **)(a3 + 8);
  v9 = (__int64)&v8[*(unsigned int *)(a3 + 20)];
  LODWORD(v10) = *(_DWORD *)(a3 + 20);
  v11 = v8;
  if ( v8 == (__int64 *)v9 )
    goto LABEL_36;
  while ( a2 != *v11 )
  {
    if ( (__int64 *)v9 == ++v11 )
      goto LABEL_8;
  }
  return a1;
}
