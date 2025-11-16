// Function: sub_BD8AF0
// Address: 0xbd8af0
//
__int64 __fastcall sub_BD8AF0(__int64 a1, _BYTE *a2, unsigned __int64 a3, __int64 a4)
{
  size_t v6; // r12
  int v7; // eax
  size_t v8; // rcx
  unsigned int v9; // eax
  unsigned int v10; // r8d
  _QWORD *v11; // rcx
  __int64 v12; // r12
  __int64 v14; // rax
  unsigned int v15; // r8d
  _QWORD *v16; // rcx
  _QWORD *v17; // rbx
  __int64 *v18; // rax
  _BYTE *v19; // rdi
  _QWORD *v20; // [rsp+0h] [rbp-160h]
  unsigned int v21; // [rsp+8h] [rbp-158h]
  _BYTE *v22; // [rsp+10h] [rbp-150h] BYREF
  size_t v23; // [rsp+18h] [rbp-148h]
  __int64 v24; // [rsp+20h] [rbp-140h]
  _BYTE dest[312]; // [rsp+28h] [rbp-138h] BYREF

  v6 = a3;
  v7 = *(_DWORD *)(a1 + 24);
  if ( v7 >= 0 )
  {
    v8 = v7;
    if ( v7 < a3 )
    {
      v6 = 1;
      if ( v7 > 1 )
      {
        if ( a3 <= v7 )
          v8 = a3;
        v6 = v8;
      }
    }
  }
  v22 = a2;
  v23 = v6;
  v9 = sub_C92610(a2, v6);
  v10 = sub_C92740(a1, a2, v6, v9);
  v11 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v10);
  if ( *v11 )
  {
    if ( *v11 != -8 )
    {
      v23 = 0;
      v22 = dest;
      v24 = 256;
      if ( v6 > 0x100 )
      {
        sub_C8D290(&v22, dest, v6, 1);
        v19 = &v22[v23];
      }
      else
      {
        if ( !v6 )
          goto LABEL_7;
        v19 = dest;
      }
      memcpy(v19, a2, v6);
      v6 += v23;
LABEL_7:
      v23 = v6;
      v12 = sub_BD8570(a1, a4, &v22);
      if ( v22 != dest )
        _libc_free(v22, a4);
      return v12;
    }
    --*(_DWORD *)(a1 + 16);
  }
  v20 = v11;
  v21 = v10;
  v14 = sub_C7D670(v6 + 17, 8);
  v15 = v21;
  v16 = v20;
  v17 = (_QWORD *)v14;
  if ( v6 )
  {
    memcpy((void *)(v14 + 16), a2, v6);
    v15 = v21;
    v16 = v20;
  }
  *((_BYTE *)v17 + v6 + 16) = 0;
  *v17 = v6;
  v17[1] = a4;
  *v16 = v17;
  ++*(_DWORD *)(a1 + 12);
  v18 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0(a1, v15));
  v12 = *v18;
  if ( *v18 == -8 || !v12 )
  {
    do
    {
      do
      {
        v12 = v18[1];
        ++v18;
      }
      while ( !v12 );
    }
    while ( v12 == -8 );
  }
  return v12;
}
