// Function: sub_2B68480
// Address: 0x2b68480
//
unsigned __int64 __fastcall sub_2B68480(__int64 a1, unsigned int a2)
{
  int v2; // eax
  __int64 v3; // r12
  unsigned __int64 v4; // rax
  unsigned __int8 **v5; // r13
  unsigned __int8 *v6; // r9
  int v7; // eax
  unsigned __int64 v8; // r10
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v14; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v15; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v16; // [rsp+18h] [rbp-98h]
  unsigned __int64 v17; // [rsp+38h] [rbp-78h]
  int v18; // [rsp+40h] [rbp-70h]
  unsigned int v19; // [rsp+44h] [rbp-6Ch]
  int v20; // [rsp+48h] [rbp-68h]
  bool v21; // [rsp+4Fh] [rbp-61h]
  int v22; // [rsp+6Ch] [rbp-44h] BYREF
  unsigned __int64 v23; // [rsp+70h] [rbp-40h] BYREF
  unsigned __int64 v24; // [rsp+78h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 208);
  v22 = 0;
  v18 = v2;
  if ( !v2 )
    goto LABEL_21;
  v19 = 0;
  v21 = 1;
  v3 = 0;
  v14 = 0;
  v17 = 0;
  v20 = 0;
  do
  {
    v5 = (unsigned __int8 **)(*(_QWORD *)(*(_QWORD *)a1 + 48 * v3) + 16LL * a2);
    v6 = *v5;
    v19 -= (*((_BYTE *)v5 + 8) == 0) - 1;
    v7 = **v5;
    v8 = (unsigned __int64)*v5;
    if ( (unsigned __int8)v7 <= 0x1Cu )
      goto LABEL_4;
    if ( v17 )
    {
      v9 = *(__int64 **)(a1 + 216);
      v24 = (unsigned __int64)*v5;
      v15 = v24;
      v16 = v6;
      v23 = v17;
      v10 = sub_2B5F980((__int64 *)&v23, 2u, v9);
      v8 = v15;
      if ( v10 && v11 && *((_QWORD *)v16 + 5) == v14 )
      {
        ++v20;
        v7 = **v5;
        goto LABEL_4;
      }
      v6 = *v5;
    }
    if ( v20 )
    {
      --v20;
    }
    else
    {
      v17 = v8;
      v20 = 1;
      v14 = *(_QWORD *)(v8 + 40);
    }
    v7 = *v6;
LABEL_4:
    v4 = 0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * (unsigned int)((v3 + 1) * (v7 + 1)));
    v23 = 0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * ((v4 >> 47) ^ v4)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v4 >> 47) ^ v4)));
    v22 = sub_C4ECF0(&v22, (__int64 *)&v23);
    if ( v21 )
      v21 = (unsigned int)**v5 - 12 <= 1;
    ++v3;
  }
  while ( v18 != v3 );
  if ( v21 )
  {
LABEL_21:
    LODWORD(v24) = 0;
    return 0xFFFFFFFFLL;
  }
  LODWORD(v24) = v22;
  v12 = v18 - v19;
  if ( v18 - v19 < v19 )
    v12 = v19;
  return __PAIR64__(v20, v12);
}
