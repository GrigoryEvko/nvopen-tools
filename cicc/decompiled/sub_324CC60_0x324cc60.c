// Function: sub_324CC60
// Address: 0x324cc60
//
void __fastcall sub_324CC60(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 v5; // rcx
  __int64 *v6; // rbx
  __int64 *v7; // r15
  __int64 v8; // r12
  const void *v9; // rax
  size_t v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdi
  const void *v13; // rax
  size_t v14; // rdx
  __int64 v15; // r9
  unsigned __int8 v16; // al
  __int64 v17; // r14
  unsigned __int8 *v18; // rax
  __int64 *i; // [rsp+0h] [rbp-40h]

  if ( a3 )
  {
    v3 = *(_BYTE *)(a3 - 16);
    if ( (v3 & 2) != 0 )
    {
      v6 = *(__int64 **)(a3 - 32);
      v5 = *(unsigned int *)(a3 - 24);
    }
    else
    {
      v5 = (*(_WORD *)(a3 - 16) >> 6) & 0xF;
      v6 = (__int64 *)(a3 - 16 - 8LL * ((v3 >> 2) & 0xF));
    }
    for ( i = &v6[v5]; i != v6; ++v6 )
    {
      v15 = *v6;
      v16 = *(_BYTE *)(*v6 - 16);
      if ( (v16 & 2) != 0 )
        v7 = *(__int64 **)(v15 - 32);
      else
        v7 = (__int64 *)(v15 + -16 - 8LL * ((v16 >> 2) & 0xF));
      v17 = *v7;
      v8 = sub_324C6D0(a1, 24576, a2, 0);
      v9 = (const void *)sub_B91420(v17);
      sub_324AD70(a1, v8, 3, v9, v10);
      v12 = v7[1];
      if ( *(_BYTE *)v12 )
      {
        if ( *(_BYTE *)v12 == 1 )
        {
          v18 = sub_AD8340(*(unsigned __int8 **)(v12 + 136), v8, v11);
          sub_324A2D0(a1, v8, (__int64)v18, 1);
        }
      }
      else
      {
        v13 = (const void *)sub_B91420(v12);
        sub_324AD70(a1, v8, 28, v13, v14);
      }
    }
  }
}
