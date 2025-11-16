// Function: sub_AC72C0
// Address: 0xac72c0
//
__int64 __fastcall sub_AC72C0(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rbx
  __int64 v3; // rcx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r9
  __int64 i; // r13
  __int64 v9; // r10
  __int64 v10; // rax
  unsigned int v11; // r13d
  int v12; // ebx
  __int64 result; // rax
  __int64 *v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  int v17; // esi
  __int64 v18; // [rsp+0h] [rbp-190h]
  __int64 v19; // [rsp+8h] [rbp-188h]
  __int64 v20; // [rsp+18h] [rbp-178h]
  int v21; // [rsp+2Ch] [rbp-164h] BYREF
  __int64 v22[4]; // [rsp+30h] [rbp-160h] BYREF
  __int64 *v23; // [rsp+50h] [rbp-140h] BYREF
  __int64 v24; // [rsp+58h] [rbp-138h]
  _BYTE v25[304]; // [rsp+60h] [rbp-130h] BYREF

  v1 = ***(_QWORD ***)(a1 + 8);
  v2 = *(unsigned int *)(v1 + 1768);
  v20 = *(_QWORD *)(v1 + 1752);
  if ( !(_DWORD)v2 )
  {
LABEL_18:
    result = v20;
    v14 = (__int64 *)(v20 + 8 * v2);
    goto LABEL_14;
  }
  v3 = 0;
  v4 = *(_DWORD *)(a1 + 4);
  v23 = (__int64 *)v25;
  v24 = 0x2000000000LL;
  v5 = 0;
  v6 = v4 & 0x7FFFFFF;
  if ( (unsigned int)v6 > 0x20uLL )
  {
    sub_C8D5F0(&v23, v25, (unsigned int)v6, 8);
    v3 = (unsigned int)v24;
    v6 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    v5 = (unsigned int)v24;
  }
  if ( (_DWORD)v6 )
  {
    v7 = (unsigned int)(v6 - 1);
    for ( i = 0; ; ++i )
    {
      v9 = *(_QWORD *)(a1 + 32 * (i - v6));
      if ( v5 + 1 > (unsigned __int64)HIDWORD(v24) )
      {
        v18 = v7;
        v19 = *(_QWORD *)(a1 + 32 * (i - v6));
        sub_C8D5F0(&v23, v25, v5 + 1, 8);
        v5 = (unsigned int)v24;
        v7 = v18;
        v9 = v19;
      }
      v23[v5] = v9;
      v5 = (unsigned int)(v24 + 1);
      LODWORD(v24) = v24 + 1;
      if ( v7 == i )
        break;
      v6 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    }
    v3 = (unsigned int)v5;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v22[2] = v3;
  v22[1] = (__int64)v23;
  v22[0] = v10;
  v21 = sub_AC5F60(v23, (__int64)&v23[v3]);
  v11 = sub_AC7240(v22, &v21);
  if ( v23 != (__int64 *)v25 )
    _libc_free(v23, &v21);
  v12 = v2 - 1;
  result = v12 & v11;
  v14 = (__int64 *)(v20 + 8 * result);
  v15 = *v14;
  if ( a1 != *v14 )
  {
    v16 = 1;
    while ( v15 != -4096 )
    {
      v17 = v16 + 1;
      result = v12 & (unsigned int)(v16 + result);
      v14 = (__int64 *)(v20 + 8LL * (unsigned int)result);
      v15 = *v14;
      if ( a1 == *v14 )
        goto LABEL_14;
      v16 = v17;
    }
    v2 = *(unsigned int *)(v1 + 1768);
    v20 = *(_QWORD *)(v1 + 1752);
    goto LABEL_18;
  }
LABEL_14:
  *v14 = -8192;
  --*(_DWORD *)(v1 + 1760);
  ++*(_DWORD *)(v1 + 1764);
  return result;
}
