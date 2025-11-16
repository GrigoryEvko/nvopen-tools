// Function: sub_AC7D40
// Address: 0xac7d40
//
__int64 __fastcall sub_AC7D40(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // r14
  __int64 v3; // r15
  __int64 v4; // r12
  __int64 *v5; // r8
  __int64 v6; // r9
  __int64 *v7; // rbx
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // ebx
  int v12; // r15d
  __int64 **v13; // rdx
  __int64 *v14; // rcx
  int v15; // edx
  int v16; // esi
  __int64 v17; // [rsp+8h] [rbp-188h]
  __int64 *v18; // [rsp+10h] [rbp-180h]
  __int64 *v19; // [rsp+18h] [rbp-178h]
  int v20; // [rsp+2Ch] [rbp-164h] BYREF
  __int64 v21[4]; // [rsp+30h] [rbp-160h] BYREF
  __int64 *v22; // [rsp+50h] [rbp-140h] BYREF
  __int64 v23; // [rsp+58h] [rbp-138h]
  _BYTE v24[304]; // [rsp+60h] [rbp-130h] BYREF

  result = *(_QWORD *)a1[1];
  v2 = *(_QWORD *)result;
  v3 = *(unsigned int *)(*(_QWORD *)result + 2112LL);
  v4 = *(_QWORD *)(*(_QWORD *)result + 2096LL);
  if ( !(_DWORD)v3 )
  {
LABEL_14:
    v13 = (__int64 **)(v4 + 8 * v3);
    goto LABEL_10;
  }
  v5 = (__int64 *)v24;
  v6 = *(a1 - 16);
  v7 = a1 - 12;
  v22 = (__int64 *)v24;
  v8 = (__int64 *)v24;
  v23 = 0x2000000000LL;
  v9 = 0;
  while ( 1 )
  {
    v8[v9] = v6;
    v9 = (unsigned int)(v23 + 1);
    LODWORD(v23) = v23 + 1;
    if ( a1 == v7 )
      break;
    v6 = *v7;
    if ( v9 + 1 > (unsigned __int64)HIDWORD(v23) )
    {
      v17 = *v7;
      v18 = v5;
      sub_C8D5F0(&v22, v5, v9 + 1, 8);
      v9 = (unsigned int)v23;
      v6 = v17;
      v5 = v18;
    }
    v8 = v22;
    v7 += 4;
  }
  v10 = a1[1];
  v19 = v5;
  v21[2] = v9;
  v21[0] = v10;
  v21[1] = (__int64)v22;
  v20 = sub_AC5F60(v22, (__int64)&v22[v9]);
  v11 = sub_AC7AE0(v21, &v20);
  if ( v22 != v19 )
    _libc_free(v22, &v20);
  v12 = v3 - 1;
  result = v12 & v11;
  v13 = (__int64 **)(v4 + 8 * result);
  v14 = *v13;
  if ( a1 != *v13 )
  {
    v15 = 1;
    while ( v14 != (__int64 *)-4096LL )
    {
      v16 = v15 + 1;
      result = v12 & (unsigned int)(v15 + result);
      v13 = (__int64 **)(v4 + 8LL * (unsigned int)result);
      v14 = *v13;
      if ( a1 == *v13 )
        goto LABEL_10;
      v15 = v16;
    }
    v4 = *(_QWORD *)(v2 + 2096);
    v3 = *(unsigned int *)(v2 + 2112);
    goto LABEL_14;
  }
LABEL_10:
  *v13 = (__int64 *)-8192LL;
  --*(_DWORD *)(v2 + 2104);
  ++*(_DWORD *)(v2 + 2108);
  return result;
}
