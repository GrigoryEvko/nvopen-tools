// Function: sub_2C75D80
// Address: 0x2c75d80
//
__int64 __fastcall sub_2C75D80(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // r15
  __int64 v5; // r13
  size_t v6; // r14
  _DWORD *v7; // rbx
  size_t v8; // r12
  size_t v9; // rdx
  int v10; // eax
  size_t v11; // r12
  size_t v12; // rdx
  int v13; // eax
  __int64 v14; // r9
  unsigned int v15; // r12d
  void *s2; // [rsp+10h] [rbp-50h] BYREF
  size_t v18; // [rsp+18h] [rbp-48h]
  _BYTE v19[64]; // [rsp+20h] [rbp-40h] BYREF

  v2 = (_BYTE *)sub_C80C60(a1, a2, 0);
  if ( v2 )
  {
    s2 = v19;
    sub_2C75590((__int64 *)&s2, v2, (__int64)&v2[v3]);
    v4 = s2;
    v5 = *(_QWORD *)&dword_5011068[2];
    if ( !*(_QWORD *)&dword_5011068[2] )
      goto LABEL_24;
  }
  else
  {
    v4 = v19;
    v18 = 0;
    v5 = *(_QWORD *)&dword_5011068[2];
    s2 = v19;
    v19[0] = 0;
    if ( !*(_QWORD *)&dword_5011068[2] )
      return 0;
  }
  v6 = v18;
  v7 = dword_5011068;
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v5 + 40);
      v9 = v6;
      if ( v8 <= v6 )
        v9 = *(_QWORD *)(v5 + 40);
      if ( v9 )
      {
        v10 = memcmp(*(const void **)(v5 + 32), v4, v9);
        if ( v10 )
          break;
      }
      if ( (__int64)(v8 - v6) >= 0x80000000LL )
        goto LABEL_13;
      if ( (__int64)(v8 - v6) > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v10 = v8 - v6;
        break;
      }
LABEL_4:
      v5 = *(_QWORD *)(v5 + 24);
      if ( !v5 )
        goto LABEL_14;
    }
    if ( v10 < 0 )
      goto LABEL_4;
LABEL_13:
    v7 = (_DWORD *)v5;
    v5 = *(_QWORD *)(v5 + 16);
  }
  while ( v5 );
LABEL_14:
  if ( v7 == dword_5011068 )
    goto LABEL_24;
  v11 = *((_QWORD *)v7 + 5);
  v12 = v6;
  if ( v11 <= v6 )
    v12 = *((_QWORD *)v7 + 5);
  if ( v12 && (v13 = memcmp(v4, *((const void **)v7 + 4), v12)) != 0 )
  {
LABEL_22:
    if ( v13 < 0 )
      goto LABEL_24;
    v15 = 1;
  }
  else
  {
    v14 = v6 - v11;
    v15 = 1;
    if ( v14 <= 0x7FFFFFFF )
    {
      if ( v14 >= (__int64)0xFFFFFFFF80000000LL )
      {
        v13 = v14;
        goto LABEL_22;
      }
LABEL_24:
      v15 = 0;
    }
  }
  if ( v4 != (_QWORD *)v19 )
    j_j___libc_free_0((unsigned __int64)v4);
  return v15;
}
