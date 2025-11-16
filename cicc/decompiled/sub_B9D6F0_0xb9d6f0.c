// Function: sub_B9D6F0
// Address: 0xb9d6f0
//
void __fastcall sub_B9D6F0(__int64 a1, __int64 a2, size_t a3)
{
  const void *v3; // r14
  __int64 v5; // rax
  unsigned __int8 v6; // dl
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 *v9; // rbx
  __int64 *v10; // r15
  __int64 v11; // r9
  const void *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  _BYTE *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r12
  unsigned int v21; // ebx
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // [rsp+10h] [rbp-80h]
  __int64 v25; // [rsp+10h] [rbp-80h]
  __int64 v26; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v27; // [rsp+30h] [rbp-60h] BYREF
  __int64 v28; // [rsp+38h] [rbp-58h]
  _BYTE v29[80]; // [rsp+40h] [rbp-50h] BYREF

  v3 = (const void *)a2;
  v27 = (__int64 *)v29;
  v28 = 0x400000000LL;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    a2 = 30;
    v5 = sub_B91C10(a1, 30);
    if ( v5 )
    {
      v6 = *(_BYTE *)(v5 - 16);
      if ( (v6 & 2) != 0 )
      {
        v8 = *(_QWORD *)(v5 - 32);
        v7 = *(unsigned int *)(v5 - 24);
      }
      else
      {
        v7 = (*(_WORD *)(v5 - 16) >> 6) & 0xF;
        v8 = v5 - 16 - 8LL * ((v6 >> 2) & 0xF);
      }
      v9 = (__int64 *)(v8 + 8 * v7);
      if ( (__int64 *)v8 != v9 )
      {
        v10 = (__int64 *)v8;
        do
        {
          v11 = *v10;
          if ( !*(_BYTE *)*v10 )
          {
            v12 = (const void *)sub_B91420(*v10);
            if ( v13 == a3 )
            {
              if ( !a3 )
                goto LABEL_18;
              a2 = (__int64)v3;
              if ( !memcmp(v12, v3, a3) )
                goto LABEL_18;
            }
            v11 = *v10;
          }
          v14 = (unsigned int)v28;
          v15 = (unsigned int)v28 + 1LL;
          if ( v15 > HIDWORD(v28) )
          {
            a2 = (__int64)v29;
            v24 = v11;
            sub_C8D5F0(&v27, v29, v15, 8);
            v14 = (unsigned int)v28;
            v11 = v24;
          }
          ++v10;
          v27[v14] = v11;
          LODWORD(v28) = v28 + 1;
        }
        while ( v9 != v10 );
      }
    }
  }
  v16 = sub_BD5C60(a1, a2);
  v17 = v3;
  v26 = v16;
  v18 = sub_B8C130(&v26, (__int64)v3, a3);
  v19 = (unsigned int)v28;
  if ( (unsigned __int64)(unsigned int)v28 + 1 > HIDWORD(v28) )
  {
    v17 = v29;
    v25 = v18;
    sub_C8D5F0(&v27, v29, (unsigned int)v28 + 1LL, 8);
    v19 = (unsigned int)v28;
    v18 = v25;
  }
  v27[v19] = v18;
  v20 = v27;
  LODWORD(v28) = v28 + 1;
  v21 = v28;
  v22 = (__int64 *)sub_BD5C60(a1, v17);
  v23 = sub_B9C770(v22, v20, (__int64 *)v21, 0, 1);
  a2 = 30;
  sub_B99FD0(a1, 0x1Eu, v23);
LABEL_18:
  if ( v27 != (__int64 *)v29 )
    _libc_free(v27, a2);
}
