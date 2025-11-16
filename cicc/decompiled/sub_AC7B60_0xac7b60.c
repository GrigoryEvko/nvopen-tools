// Function: sub_AC7B60
// Address: 0xac7b60
//
void __fastcall sub_AC7B60(__int64 a1, __int64 *a2)
{
  __int64 v2; // r15
  __int64 v3; // r12
  __int64 *v4; // r8
  __int64 v5; // r9
  __int64 *v7; // rbx
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // ebx
  int v12; // r15d
  unsigned int v13; // eax
  __int64 **v14; // rdx
  __int64 *v15; // rcx
  int v16; // edx
  int v17; // esi
  __int64 v18; // [rsp+8h] [rbp-188h]
  __int64 *v19; // [rsp+10h] [rbp-180h]
  __int64 *v20; // [rsp+18h] [rbp-178h]
  int v21; // [rsp+2Ch] [rbp-164h] BYREF
  __int64 v22[4]; // [rsp+30h] [rbp-160h] BYREF
  __int64 *v23; // [rsp+50h] [rbp-140h] BYREF
  __int64 v24; // [rsp+58h] [rbp-138h]
  _BYTE v25[304]; // [rsp+60h] [rbp-130h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)v2 )
  {
LABEL_14:
    v14 = (__int64 **)(v3 + 8 * v2);
    goto LABEL_10;
  }
  v4 = (__int64 *)v25;
  v5 = *(a2 - 16);
  v7 = a2 - 12;
  v23 = (__int64 *)v25;
  v8 = (__int64 *)v25;
  v24 = 0x2000000000LL;
  v9 = 0;
  while ( 1 )
  {
    v8[v9] = v5;
    v9 = (unsigned int)(v24 + 1);
    LODWORD(v24) = v24 + 1;
    if ( a2 == v7 )
      break;
    v5 = *v7;
    if ( v9 + 1 > (unsigned __int64)HIDWORD(v24) )
    {
      v18 = *v7;
      v19 = v4;
      sub_C8D5F0(&v23, v4, v9 + 1, 8);
      v9 = (unsigned int)v24;
      v5 = v18;
      v4 = v19;
    }
    v8 = v23;
    v7 += 4;
  }
  v10 = a2[1];
  v20 = v4;
  v22[2] = v9;
  v22[0] = v10;
  v22[1] = (__int64)v23;
  v21 = sub_AC5F60(v23, (__int64)&v23[v9]);
  v11 = sub_AC7AE0(v22, &v21);
  if ( v23 != v20 )
    _libc_free(v23, &v21);
  v12 = v2 - 1;
  v13 = v12 & v11;
  v14 = (__int64 **)(v3 + 8LL * (v12 & v11));
  v15 = *v14;
  if ( a2 != *v14 )
  {
    v16 = 1;
    while ( v15 != (__int64 *)-4096LL )
    {
      v17 = v16 + 1;
      v13 = v12 & (v16 + v13);
      v14 = (__int64 **)(v3 + 8LL * v13);
      v15 = *v14;
      if ( a2 == *v14 )
        goto LABEL_10;
      v16 = v17;
    }
    v3 = *(_QWORD *)(a1 + 8);
    v2 = *(unsigned int *)(a1 + 24);
    goto LABEL_14;
  }
LABEL_10:
  *v14 = (__int64 *)-8192LL;
  --*(_DWORD *)(a1 + 16);
  ++*(_DWORD *)(a1 + 20);
}
