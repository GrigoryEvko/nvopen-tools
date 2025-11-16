// Function: sub_980380
// Address: 0x980380
//
__int64 __fastcall sub_980380(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // r15
  size_t *v10; // rcx
  __int64 *v11; // rbx
  __int64 v12; // r12
  unsigned int *v13; // r14
  _BYTE *v14; // rdi
  unsigned __int8 *v15; // r8
  size_t v16; // r13
  __int64 v17; // rax
  unsigned __int8 *v18; // [rsp+0h] [rbp-50h]
  size_t *v19; // [rsp+8h] [rbp-48h]
  size_t *v20; // [rsp+8h] [rbp-48h]
  size_t v21; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = v4 + 40 * v3;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v4 <= 0xFFFFFFFD )
        {
          v6 = *(_QWORD *)(v4 + 8);
          if ( v6 != v4 + 24 )
            break;
        }
        v4 += 40;
        if ( v5 == v4 )
          goto LABEL_7;
      }
      v7 = *(_QWORD *)(v4 + 24);
      v4 += 40;
      j_j___libc_free_0(v6, v7 + 1);
    }
    while ( v5 != v4 );
LABEL_7:
    v3 = *(unsigned int *)(a1 + 24);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 8), 40 * v3, 8);
  result = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = result;
  if ( !(_DWORD)result )
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  result = sub_C7D670(40 * result, 8);
  v9 = *(unsigned int *)(a1 + 24);
  v10 = &v21;
  *(_QWORD *)(a1 + 8) = result;
  v11 = (__int64 *)(result + 8);
  *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 16);
  *(_DWORD *)(a1 + 20) = *(_DWORD *)(a2 + 20);
  v12 = 0;
  v13 = *(unsigned int **)(a2 + 8);
  if ( v9 )
  {
    while ( 1 )
    {
      result = *v13;
      *((_DWORD *)v11 - 2) = result;
      if ( (unsigned int)result <= 0xFFFFFFFD )
        break;
LABEL_13:
      ++v12;
      v13 += 10;
      v11 += 5;
      if ( v9 == v12 )
        return result;
    }
    v14 = v11 + 2;
    *v11 = (__int64)(v11 + 2);
    v15 = (unsigned __int8 *)*((_QWORD *)v13 + 1);
    v16 = *((_QWORD *)v13 + 2);
    result = (__int64)&v15[v16];
    if ( &v15[v16] )
    {
      if ( !v15 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
    }
    v21 = *((_QWORD *)v13 + 2);
    if ( v16 > 0xF )
    {
      v18 = v15;
      v19 = v10;
      v17 = sub_22409D0(v11, v10, 0);
      v10 = v19;
      v15 = v18;
      *v11 = v17;
      v14 = (_BYTE *)v17;
      v11[2] = v21;
    }
    else
    {
      if ( v16 == 1 )
      {
        result = *v15;
        *((_BYTE *)v11 + 16) = result;
LABEL_20:
        v11[1] = v16;
        v14[v16] = 0;
        goto LABEL_13;
      }
      if ( !v16 )
        goto LABEL_20;
    }
    v20 = v10;
    result = (__int64)memcpy(v14, v15, v16);
    v16 = v21;
    v14 = (_BYTE *)*v11;
    v10 = v20;
    goto LABEL_20;
  }
  return result;
}
