// Function: sub_35B0830
// Address: 0x35b0830
//
unsigned __int64 __fastcall sub_35B0830(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 *a5, unsigned __int8 *a6)
{
  __int64 v7; // r9
  unsigned __int64 result; // rax
  int *i; // r14
  int v11; // eax
  unsigned __int8 v12; // cl
  char *v13; // rdx
  __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  int *v16; // rbx
  int v17; // r15d
  _QWORD *v18; // r15
  __int64 v19; // r14
  int *v20; // r13
  _QWORD *v21; // r12
  bool v22; // r9
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rdx
  __int64 *v27; // [rsp+0h] [rbp-90h]
  __int64 v28; // [rsp+8h] [rbp-88h]
  int *v29; // [rsp+10h] [rbp-80h]
  __int64 v30; // [rsp+18h] [rbp-78h]
  __int64 *v31; // [rsp+20h] [rbp-70h]
  char v32; // [rsp+20h] [rbp-70h]
  __int64 *v33; // [rsp+20h] [rbp-70h]
  __int64 *v34; // [rsp+20h] [rbp-70h]
  __int64 v35; // [rsp+28h] [rbp-68h]
  _QWORD *v36; // [rsp+28h] [rbp-68h]
  __int64 v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+28h] [rbp-68h]
  int *v40; // [rsp+40h] [rbp-50h]
  char v42; // [rsp+5Bh] [rbp-35h] BYREF
  int v43[13]; // [rsp+5Ch] [rbp-34h] BYREF

  v7 = *(_QWORD *)(a1 + 32);
  result = v7 + 4LL * *(unsigned int *)(a1 + 40);
  v40 = (int *)result;
  if ( result != v7 )
  {
    for ( i = *(int **)(a1 + 32); v40 != i; ++i )
    {
      v11 = *i;
      v43[0] = *i;
      if ( a4 )
        *a5 += *(_QWORD *)(*(_QWORD *)(a3 + 8) + 40LL * (unsigned int)(*(_DWORD *)(a3 + 32) + v11) + 8);
      v12 = *(_BYTE *)(*(_QWORD *)(a3 + 8) + 40LL * (unsigned int)(*(_DWORD *)(a3 + 32) + v11) + 16);
      v13 = (char *)a6;
      if ( v12 > *a6 )
        v13 = &v42;
      v42 = *(_BYTE *)(*(_QWORD *)(a3 + 8) + 40LL * (unsigned int)(*(_DWORD *)(a3 + 32) + v11) + 16);
      *a6 = *v13;
      v14 = -(1LL << v12) & ((1LL << v12) + *a5 - 1);
      *a5 = v14;
      if ( a4 )
      {
        v7 = *(_QWORD *)(a3 + 8);
        *(_QWORD *)(v7 + 40LL * (unsigned int)(*(_DWORD *)(a3 + 32) + v11)) = -v14;
        if ( *(_QWORD *)(a2 + 120) )
          goto LABEL_17;
      }
      else
      {
        *(_QWORD *)(*(_QWORD *)(a3 + 8) + 40LL * (unsigned int)(*(_DWORD *)(a3 + 32) + v11)) = v14;
        *a5 = *(_QWORD *)(*(_QWORD *)(a3 + 8) + 40LL * (unsigned int)(*(_DWORD *)(a3 + 32) + v11) + 8) + v14;
        if ( *(_QWORD *)(a2 + 120) )
        {
LABEL_17:
          v31 = a5;
          v35 = a2;
          result = sub_BB8160(a2 + 80, v43);
          a2 = v35;
          a5 = v31;
          continue;
        }
      }
      v15 = *(unsigned int *)(a2 + 8);
      v16 = (int *)(*(_QWORD *)a2 + 4 * v15);
      if ( *(int **)a2 == v16 )
      {
        if ( v15 <= 0xF )
        {
          v17 = v43[0];
LABEL_29:
          result = *(unsigned int *)(a2 + 12);
          v26 = v15 + 1;
          if ( v26 > result )
          {
            v34 = a5;
            v38 = a2;
            sub_C8D5F0(a2, (const void *)(a2 + 16), v26, 4u, (__int64)a5, v7);
            a2 = v38;
            a5 = v34;
            result = *(_QWORD *)v38;
            v16 = (int *)(*(_QWORD *)v38 + 4LL * *(unsigned int *)(v38 + 8));
          }
          *v16 = v17;
          ++*(_DWORD *)(a2 + 8);
          continue;
        }
        v30 = a2 + 80;
      }
      else
      {
        v17 = v43[0];
        result = *(_QWORD *)a2;
        while ( *(_DWORD *)result != v43[0] )
        {
          result += 4LL;
          if ( v16 == (int *)result )
            goto LABEL_18;
        }
        if ( v16 != (int *)result )
          continue;
LABEL_18:
        if ( v15 <= 0xF )
          goto LABEL_29;
        v29 = i;
        v18 = (_QWORD *)(a2 + 88);
        v19 = a2;
        v28 = a3;
        v20 = *(int **)a2;
        v21 = (_QWORD *)(a2 + 80);
        v30 = a2 + 80;
        v27 = a5;
        do
        {
          v24 = sub_BB8210(v21, (__int64)v18, v20);
          if ( v25 )
          {
            v22 = v24 || v18 == (_QWORD *)v25 || *v20 < *(_DWORD *)(v25 + 32);
            v32 = v22;
            v36 = (_QWORD *)v25;
            v23 = sub_22077B0(0x28u);
            *(_DWORD *)(v23 + 32) = *v20;
            sub_220F040(v32, v23, v36, v18);
            ++*(_QWORD *)(v19 + 120);
          }
          ++v20;
        }
        while ( v16 != v20 );
        a2 = v19;
        a3 = v28;
        i = v29;
        a5 = v27;
      }
      *(_DWORD *)(a2 + 8) = 0;
      v33 = a5;
      v37 = a2;
      result = sub_BB8160(v30, v43);
      a5 = v33;
      a2 = v37;
    }
  }
  return result;
}
