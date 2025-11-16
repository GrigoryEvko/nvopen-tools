// Function: sub_15CF2F0
// Address: 0x15cf2f0
//
__int64 __fastcall sub_15CF2F0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rbx
  int v8; // r15d
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // r8
  __int64 *v13; // rbx
  char *v14; // rcx
  char *v15; // rax
  signed __int64 v16; // r15
  __int64 v17; // rax
  __int64 *v18; // rbx
  __int64 *i; // r14
  __int64 v20; // r15
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  _DWORD *v26; // rdx
  __int64 v27; // rdi
  __int64 *v29; // [rsp+0h] [rbp-D0h]
  void *src; // [rsp+10h] [rbp-C0h]
  char *srca; // [rsp+10h] [rbp-C0h]
  _DWORD *srcb; // [rsp+10h] [rbp-C0h]
  __int64 *v33; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v34; // [rsp+28h] [rbp-A8h] BYREF
  __int64 *v35; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+40h] [rbp-90h]
  __int64 *v37; // [rsp+50h] [rbp-80h] BYREF
  __int64 v38; // [rsp+58h] [rbp-78h]
  _BYTE v39[112]; // [rsp+60h] [rbp-70h] BYREF

  v5 = *(_QWORD *)(*a3 + 8);
  src = (void *)*a3;
  if ( !v5 )
  {
LABEL_46:
    v37 = (__int64 *)v39;
    v38 = 0x800000000LL;
    if ( !a2 )
    {
      v29 = (__int64 *)v39;
      goto LABEL_42;
    }
    goto LABEL_16;
  }
  while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v5) + 16) - 25) > 9u )
  {
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      goto LABEL_46;
  }
  v6 = v5;
  v7 = 0;
  v37 = (__int64 *)v39;
  v38 = 0x800000000LL;
  while ( 1 )
  {
    v6 = *(_QWORD *)(v6 + 8);
    if ( !v6 )
      break;
    while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v6) + 16) - 25) <= 9u )
    {
      v6 = *(_QWORD *)(v6 + 8);
      ++v7;
      if ( !v6 )
        goto LABEL_7;
    }
  }
LABEL_7:
  v8 = v7 + 1;
  if ( v7 + 1 > 8 )
  {
    sub_16CD150(&v37, v39, v7 + 1, 8);
    v9 = &v37[(unsigned int)v38];
  }
  else
  {
    v9 = (__int64 *)v39;
  }
  v10 = sub_1648700(v5);
LABEL_12:
  if ( v9 )
    *v9 = *(_QWORD *)(v10 + 40);
  while ( 1 )
  {
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      break;
    v10 = sub_1648700(v5);
    if ( (unsigned __int8)(*(_BYTE *)(v10 + 16) - 25) <= 9u )
    {
      ++v9;
      goto LABEL_12;
    }
  }
  v11 = (unsigned int)(v38 + v8);
  LODWORD(v38) = v38 + v8;
  if ( a2 )
  {
LABEL_16:
    sub_15CE790(&v35, (__int64 *)(a2 + 112), (__int64)src);
    if ( v36 != *(_QWORD *)(a2 + 120) + 56LL * *(unsigned int *)(a2 + 136) )
    {
      v12 = *(__int64 **)(v36 + 8);
      v33 = &v12[*(unsigned int *)(v36 + 16)];
      if ( v12 != v33 )
      {
        v13 = *(__int64 **)(v36 + 8);
        do
        {
          while ( 1 )
          {
            v17 = *v13;
            v34 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v17 & 4) == 0 )
              break;
            ++v13;
            sub_15CDD90((__int64)&v37, &v34);
            if ( v33 == v13 )
              goto LABEL_24;
          }
          srca = (char *)&v37[(unsigned int)v38];
          v14 = (char *)sub_15CF090(v37, (__int64)srca, (__int64 *)&v34);
          v15 = (char *)v37;
          v16 = (char *)&v37[(unsigned int)v38] - srca;
          if ( srca != (char *)&v37[(unsigned int)v38] )
          {
            v14 = (char *)memmove(v14, srca, (char *)&v37[(unsigned int)v38] - srca);
            v15 = (char *)v37;
          }
          ++v13;
          LODWORD(v38) = (&v14[v16] - v15) >> 3;
        }
        while ( v33 != v13 );
      }
    }
LABEL_24:
    v11 = (unsigned int)v38;
  }
  v18 = &v37[v11];
  v29 = v37;
  if ( v18 != v37 )
  {
    for ( i = v37; v18 != i; ++i )
    {
      v20 = *i;
      v21 = sub_15CC510(a1, *i);
      if ( !v21 )
        continue;
      v22 = *a3;
      v23 = *(_QWORD *)(*(_QWORD *)(*a3 + 56) + 80LL);
      if ( v23 )
      {
        v24 = v23 - 24;
        if ( v22 == v24 )
          continue;
        if ( v20 == v24 )
          goto LABEL_40;
      }
      else if ( !v20 )
      {
LABEL_36:
        if ( v29 != (__int64 *)v39 )
          _libc_free((unsigned __int64)v29);
        return 1;
      }
      srcb = (_DWORD *)v21;
      v25 = sub_15CC510(a1, v22);
      v26 = srcb;
      if ( !v25 )
        goto LABEL_36;
      while ( (_DWORD *)v25 != v26 )
      {
        if ( *(_DWORD *)(v25 + 16) < v26[4] )
        {
          v27 = v25;
          v25 = (__int64)v26;
          v26 = (_DWORD *)v27;
        }
        v25 = *(_QWORD *)(v25 + 8);
        if ( !v25 )
          goto LABEL_36;
      }
      v24 = *(_QWORD *)v25;
LABEL_40:
      if ( v22 != v24 )
        goto LABEL_36;
    }
  }
LABEL_42:
  if ( v29 != (__int64 *)v39 )
    _libc_free((unsigned __int64)v29);
  return 0;
}
