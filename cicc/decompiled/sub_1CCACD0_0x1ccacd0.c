// Function: sub_1CCACD0
// Address: 0x1ccacd0
//
__int64 __fastcall sub_1CCACD0(__int64 a1, const void *a2, size_t a3, unsigned int a4)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  int v7; // r13d
  unsigned int i; // r15d
  __int64 v9; // rax
  __int64 v10; // r14
  _BYTE *v11; // rdi
  const void *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v19; // [rsp+8h] [rbp-68h]
  int v20; // [rsp+10h] [rbp-60h]
  char *v23; // [rsp+20h] [rbp-50h] BYREF
  __int16 v24; // [rsp+30h] [rbp-40h]

  if ( a1 )
  {
    v24 = 257;
    if ( *off_4CD4980[0] )
    {
      v23 = off_4CD4980[0];
      LOBYTE(v24) = 3;
    }
    v5 = sub_1632310(a1, (__int64)&v23);
    v6 = v5;
    if ( v5 )
    {
      v7 = sub_161F520(v5);
      if ( v7 )
      {
        for ( i = 0; v7 != i; ++i )
        {
          v9 = sub_161F530(v6, i);
          v10 = v9;
          if ( !v9 )
            continue;
          v11 = *(_BYTE **)(v9 - 8LL * *(unsigned int *)(v9 + 8));
          if ( *v11 )
            continue;
          v12 = (const void *)sub_161E970((__int64)v11);
          if ( a3 != v13 || a3 && memcmp(v12, a2, a3) )
            continue;
          v14 = *(_QWORD *)(v10 + 8 * (1LL - *(unsigned int *)(v10 + 8)));
          if ( *(_BYTE *)v14 != 1 )
            continue;
          v15 = *(_QWORD *)(v14 + 136);
          if ( *(_BYTE *)(v15 + 16) != 13 )
            continue;
          v16 = a4;
          if ( *(_DWORD *)(v15 + 32) <= 0x40u )
          {
            v17 = *(_QWORD *)(v15 + 24);
          }
          else
          {
            v19 = *(_QWORD *)(v14 + 136);
            v16 = a4;
            v20 = *(_DWORD *)(v15 + 32);
            if ( v20 - (unsigned int)sub_16A57B0(v15 + 24) > 0x40 )
              continue;
            v17 = **(_QWORD **)(v19 + 24);
          }
          if ( v16 == v17 )
            return 1;
        }
      }
    }
  }
  return 0;
}
