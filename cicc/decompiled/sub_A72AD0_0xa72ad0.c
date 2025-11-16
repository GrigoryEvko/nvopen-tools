// Function: sub_A72AD0
// Address: 0xa72ad0
//
__int64 __fastcall sub_A72AD0(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rax
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // r14
  __int64 v6; // rbx
  int v7; // r13d
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rcx
  __int64 result; // rax
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rdi
  unsigned __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r14
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rbx
  int v26; // r13d
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // rbx
  int v32; // r13d
  __int64 v33; // rax

  v2 = *(_BYTE *)(a1 + 8);
  if ( !v2 )
  {
    LODWORD(v17) = sub_A71AD0(a1);
    result = *(unsigned int *)(a2 + 8);
    v18 = result + 1;
    if ( result + 1 <= (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
LABEL_16:
      *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v17;
      ++*(_DWORD *)(a2 + 8);
      return result;
    }
LABEL_25:
    sub_C8D5F0(a2, a2 + 16, v18, 4);
    result = *(unsigned int *)(a2 + 8);
    goto LABEL_16;
  }
  if ( v2 == 1 )
  {
    v24 = sub_A71B70(a1);
    goto LABEL_20;
  }
  if ( v2 != 2 )
  {
    if ( v2 != 3 )
    {
      if ( v2 == 4 )
      {
        v31 = sub_A72A80(a1);
        v32 = sub_A71AD0(a1);
        v33 = *(unsigned int *)(a2 + 8);
        if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, a2 + 16, v33 + 1, 4);
          v33 = *(unsigned int *)(a2 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a2 + 4 * v33) = v32;
        ++*(_DWORD *)(a2 + 8);
        sub_C439F0(v31, a2);
        return sub_C439F0(v31 + 16, a2);
      }
      else
      {
        v3 = sub_A72AB0(a1);
        v5 = v4;
        v6 = v3;
        v7 = sub_A71AD0(a1);
        v8 = *(unsigned int *)(a2 + 8);
        if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, a2 + 16, v8 + 1, 4);
          v8 = *(unsigned int *)(a2 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a2 + 4 * v8) = v7;
        v9 = *(unsigned int *)(a2 + 12);
        v10 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v10;
        if ( v10 + 1 > v9 )
        {
          sub_C8D5F0(a2, a2 + 16, v10 + 1, 4);
          v10 = *(unsigned int *)(a2 + 8);
        }
        v11 = HIDWORD(v5);
        *(_DWORD *)(*(_QWORD *)a2 + 4 * v10) = v5;
        v12 = *(unsigned int *)(a2 + 12);
        result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = result;
        if ( result + 1 > v12 )
        {
          sub_C8D5F0(a2, a2 + 16, result + 1, 4);
          result = *(unsigned int *)(a2 + 8);
        }
        v14 = 32 * v5;
        *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v11;
        v15 = v6 + v14;
        ++*(_DWORD *)(a2 + 8);
        if ( v6 + v14 != v6 )
        {
          do
          {
            sub_C439F0(v6, a2);
            v16 = v6 + 16;
            v6 += 32;
            result = sub_C439F0(v16, a2);
          }
          while ( v15 != v6 );
        }
      }
      return result;
    }
    v24 = sub_A72A50(a1);
LABEL_20:
    v25 = v24;
    v26 = sub_A71AD0(a1);
    v27 = *(unsigned int *)(a2 + 8);
    if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, a2 + 16, v27 + 1, 4);
      v27 = *(unsigned int *)(a2 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * v27) = v26;
    v28 = *(unsigned int *)(a2 + 12);
    v29 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v29;
    if ( v29 + 1 > v28 )
    {
      sub_C8D5F0(a2, a2 + 16, v29 + 1, 4);
      v29 = *(unsigned int *)(a2 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * v29) = v25;
    v17 = HIDWORD(v25);
    v30 = *(unsigned int *)(a2 + 12);
    result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    v18 = result + 1;
    *(_DWORD *)(a2 + 8) = result;
    if ( result + 1 <= v30 )
      goto LABEL_16;
    goto LABEL_25;
  }
  v19 = sub_A72230(a1);
  v21 = v20;
  v22 = v19;
  v23 = sub_A71FC0(a1);
  result = sub_C653C0(a2, v23);
  if ( v21 )
    return sub_C653C0(a2, v22);
  return result;
}
