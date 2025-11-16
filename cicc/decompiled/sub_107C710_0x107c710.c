// Function: sub_107C710
// Address: 0x107c710
//
char __fastcall sub_107C710(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 i; // rcx
  unsigned __int64 v10; // rdx
  _BYTE *v11; // rax
  __int64 **v12; // rax
  __int64 *v13; // rbx
  unsigned __int8 v14; // r14
  unsigned __int64 v15; // rdi
  size_t v16; // r14
  __int64 v17; // rdi
  __int64 v18; // r15
  size_t v19; // r15
  unsigned __int64 v21; // [rsp+8h] [rbp-48h]
  _QWORD v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1[1];
  v8 = 1LL << *(_BYTE *)(a2 + 32);
  i = v8 + v7 - 1;
  v10 = i & -v8;
  if ( v10 != v7 )
  {
    if ( v10 >= v7 )
    {
      if ( v10 > a1[2] )
      {
        v21 = v10;
        sub_C8D290((__int64)a1, a1 + 3, v10, 1u, a5, a6);
        v10 = v21;
        v11 = (_BYTE *)(*a1 + a1[1]);
        for ( i = v21 + *a1; (_BYTE *)i != v11; ++v11 )
        {
LABEL_5:
          if ( v11 )
            *v11 = 0;
        }
      }
      else
      {
        v11 = (_BYTE *)(*a1 + v7);
        i = v10 + *a1;
        if ( v11 != (_BYTE *)i )
          goto LABEL_5;
      }
    }
    a1[1] = v10;
  }
  v12 = *(__int64 ***)(a2 + 8);
  v13 = *v12;
  if ( *v12 )
  {
    while ( 1 )
    {
      if ( (*((_BYTE *)v13 + 29) & 1) != 0 )
        sub_C64ED0("only data supported in data sections", 1u);
      v14 = *((_BYTE *)v13 + 28);
      if ( !v14 )
        break;
      if ( v14 == 2 )
      {
        LOBYTE(v12) = sub_E81180(v13[5], v22);
        if ( !(_BYTE)v12 )
          BUG();
        v16 = v22[0] * *((unsigned __int8 *)v13 + 30);
        v17 = a1[1];
        v18 = v13[4];
        v10 = v16 + v17;
        if ( v16 + v17 > a1[2] )
        {
          LOBYTE(v12) = sub_C8D290((__int64)a1, a1 + 3, v10, 1u, a5, a6);
          v17 = a1[1];
        }
        if ( v16 )
        {
          LOBYTE(v12) = (unsigned __int8)memset((void *)(*a1 + v17), (unsigned __int8)v18, v16);
          v17 = a1[1];
        }
        a1[1] = v17 + v16;
        v13 = (__int64 *)*v13;
        if ( !v13 )
          return (char)v12;
      }
      else
      {
        LOBYTE(v12) = sub_107C6A0(a1, (__int64)(v13 + 5), v10, i, a5, a6);
LABEL_19:
        v13 = (__int64 *)*v13;
        if ( !v13 )
          return (char)v12;
      }
    }
    if ( *((_DWORD *)v13 + 10) != 1 )
      sub_C64ED0("only byte values supported for alignment", 1u);
    if ( (*((_BYTE *)v13 + 31) & 1) == 0 )
      v14 = *((_BYTE *)v13 + 32);
    i = *((unsigned __int8 *)v13 + 30);
    v15 = a1[1];
    v10 = -(1LL << i) & ((1LL << i) + v15 - 1);
    v12 = (__int64 **)(v15 + *((unsigned int *)v13 + 11));
    if ( v10 > (unsigned __int64)v12 )
      v10 = v15 + *((unsigned int *)v13 + 11);
    if ( v10 != v15 )
    {
      if ( v10 >= v15 )
      {
        v19 = v10 - v15;
        if ( v10 > a1[2] )
        {
          sub_C8D290((__int64)a1, a1 + 3, v10, 1u, a5, a6);
          v15 = a1[1];
        }
        LOBYTE(v12) = (unsigned __int8)memset((void *)(*a1 + v15), v14, v19);
        a1[1] += v19;
      }
      else
      {
        a1[1] = v10;
      }
    }
    goto LABEL_19;
  }
  return (char)v12;
}
