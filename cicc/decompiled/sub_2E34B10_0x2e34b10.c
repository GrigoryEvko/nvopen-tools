// Function: sub_2E34B10
// Address: 0x2e34b10
//
signed __int64 __fastcall sub_2E34B10(__int64 *a1, char *a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 v4; // r15
  __int64 *v6; // rbx
  __int64 v7; // r9
  __int64 *v8; // rax
  unsigned int v9; // ecx
  unsigned int v10; // edx
  unsigned int v11; // edi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  char *v16; // r14
  char *v17; // rdx
  int v18; // r8d
  unsigned __int64 v19; // r12
  unsigned int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8

  result = a2 - (char *)a1;
  if ( a2 - (char *)a1 <= 256 )
    return result;
  v4 = a3;
  v6 = a1 + 2;
  if ( !a3 )
  {
    v19 = (unsigned __int64)a2;
    goto LABEL_23;
  }
  while ( 2 )
  {
    v7 = a1[2];
    --v4;
    v8 = &a1[2 * (result >> 5)];
    v9 = *(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v7 >> 1) & 3;
    v10 = *(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v8 >> 1) & 3;
    v11 = *(_DWORD *)((*((_QWORD *)a2 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*((__int64 *)a2 - 2) >> 1) & 3;
    if ( v9 >= v10 )
    {
      if ( v9 < v11 )
        goto LABEL_6;
      if ( v10 < v11 )
      {
LABEL_17:
        v25 = *a1;
        *a1 = *((_QWORD *)a2 - 2);
        v26 = *((_QWORD *)a2 - 1);
        *((_QWORD *)a2 - 2) = v25;
        v27 = a1[1];
        a1[1] = v26;
        *((_QWORD *)a2 - 1) = v27;
        goto LABEL_7;
      }
LABEL_21:
      v28 = *a1;
      *a1 = *v8;
      v29 = v8[1];
      *v8 = v28;
      v30 = a1[1];
      a1[1] = v29;
      v8[1] = v30;
      goto LABEL_7;
    }
    if ( v10 < v11 )
      goto LABEL_21;
    if ( v9 < v11 )
      goto LABEL_17;
LABEL_6:
    v12 = *a1;
    v13 = a1[3];
    *a1 = v7;
    a1[2] = v12;
    v14 = a1[1];
    a1[1] = v13;
    a1[3] = v14;
LABEL_7:
    v15 = *a1;
    v16 = (char *)v6;
    v17 = a2;
    v18 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    while ( 1 )
    {
      v19 = (unsigned __int64)v16;
      v20 = v18 | (v15 >> 1) & 3;
      if ( (*(_DWORD *)((*(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)v16 >> 1) & 3) < v20 )
        goto LABEL_14;
      v21 = *((_QWORD *)v17 - 2);
      v17 -= 16;
      while ( v20 < (*(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v21 >> 1) & 3) )
      {
        v21 = *((_QWORD *)v17 - 2);
        v17 -= 16;
      }
      if ( v16 >= v17 )
        break;
      v22 = *(_QWORD *)v16;
      *(_QWORD *)v16 = *(_QWORD *)v17;
      v23 = *((_QWORD *)v17 + 1);
      *(_QWORD *)v17 = v22;
      v24 = *((_QWORD *)v16 + 1);
      *((_QWORD *)v16 + 1) = v23;
      *((_QWORD *)v17 + 1) = v24;
      v15 = *a1;
      v18 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24);
LABEL_14:
      v16 += 16;
    }
    sub_2E34B10(v16, a2, v4);
    result = v16 - (char *)a1;
    if ( v16 - (char *)a1 > 256 )
    {
      if ( v4 )
      {
        a2 = v16;
        continue;
      }
LABEL_23:
      sub_2E34A00(a1, (char *)v19, v19);
      do
      {
        v19 -= 16LL;
        v31 = *(_QWORD *)v19;
        v32 = *(_QWORD *)(v19 + 8);
        *(_QWORD *)v19 = *a1;
        *(_QWORD *)(v19 + 8) = a1[1];
        result = (signed __int64)sub_2E301C0((__int64)a1, 0, (__int64)(v19 - (_QWORD)a1) >> 4, v31, v32);
      }
      while ( (__int64)(v19 - (_QWORD)a1) > 16 );
    }
    return result;
  }
}
