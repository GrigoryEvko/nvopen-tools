// Function: sub_2611C00
// Address: 0x2611c00
//
_BYTE *__fastcall sub_2611C00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rcx
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rbx
  unsigned int v13; // r13d
  __int64 v14; // r15
  _BYTE *v15; // rax
  bool v16; // zf
  _QWORD **v17; // r10
  _BYTE *(__fastcall *v18)(__int64, __int64, __int64, __int64); // rax
  __int64 v19; // rdx
  _BYTE *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned int v23; // r15d
  __int64 v24; // rdx
  _BYTE *v25; // rax
  _BYTE *result; // rax
  _BYTE *v27; // rax
  _BYTE *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rax
  _WORD *v32; // rdx
  __int64 v33; // [rsp+0h] [rbp-50h]
  __int64 v34; // [rsp+8h] [rbp-48h]
  _QWORD **v35; // [rsp+10h] [rbp-40h]
  unsigned int v36; // [rsp+1Ch] [rbp-34h]
  unsigned int v37; // [rsp+1Ch] [rbp-34h]

  v6 = a3;
  v7 = a4;
  v8 = a2;
  v9 = a1;
  v34 = a3;
  v10 = *(_QWORD *)(a1 + 152);
  v11 = *(_QWORD *)(a1 + 144);
  if ( v10 != v11 )
  {
    v36 = (v10 - v11) >> 3;
    if ( v36 )
    {
      v33 = v7;
      v12 = v7;
      v13 = 0;
      v14 = v6;
      while ( 1 )
      {
        v17 = *(_QWORD ***)(v11 + 8LL * v13);
        v18 = (_BYTE *(__fastcall *)(__int64, __int64, __int64, __int64))(*v17)[3];
        if ( v18 != sub_23103E0 )
          break;
        v19 = *(_QWORD *)(a2 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v19) > 5 )
        {
          *(_DWORD *)v19 = 1668507491;
          *(_WORD *)(v19 + 4) = 10339;
          *(_QWORD *)(a2 + 32) += 6LL;
        }
        else
        {
          v35 = v17;
          sub_CB6200(a2, "cgscc(", 6u);
          v17 = v35;
        }
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*v17[1] + 24LL))(v17[1], a2, v14, v12);
        v15 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v15 < *(_QWORD *)(a2 + 24) )
        {
          *(_QWORD *)(a2 + 32) = v15 + 1;
          *v15 = 41;
          goto LABEL_7;
        }
        ++v13;
        sub_CB5D20(a2, 41);
        v16 = v36 == v13;
        if ( v36 > v13 )
        {
LABEL_31:
          v27 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v27 >= *(_QWORD *)(a2 + 24) )
          {
            sub_CB5D20(a2, 44);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = v27 + 1;
            *v27 = 44;
          }
          goto LABEL_9;
        }
LABEL_8:
        if ( v16 )
        {
          v7 = v33;
          v9 = a1;
          v8 = a2;
          goto LABEL_14;
        }
LABEL_9:
        v11 = *(_QWORD *)(a1 + 144);
      }
      v18((__int64)v17, a2, v14, v12);
LABEL_7:
      v16 = v36 == ++v13;
      if ( v36 > v13 )
        goto LABEL_31;
      goto LABEL_8;
    }
LABEL_14:
    v20 = *(_BYTE **)(v8 + 32);
    if ( (unsigned __int64)v20 >= *(_QWORD *)(v8 + 24) )
    {
      sub_CB5D20(v8, 44);
    }
    else
    {
      *(_QWORD *)(v8 + 32) = v20 + 1;
      *v20 = 44;
    }
  }
  v21 = *(_QWORD *)(v8 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v8 + 24) - v21) <= 5 )
  {
    sub_CB6200(v8, "cgscc(", 6u);
  }
  else
  {
    *(_DWORD *)v21 = 1668507491;
    *(_WORD *)(v21 + 4) = 10339;
    *(_QWORD *)(v8 + 32) += 6LL;
  }
  if ( *(_DWORD *)(v9 + 96) )
  {
    v29 = *(_QWORD *)(v8 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v8 + 24) - v29) <= 6 )
    {
      v30 = sub_CB6200(v8, "devirt<", 7u);
    }
    else
    {
      *(_DWORD *)v29 = 1769366884;
      v30 = v8;
      *(_WORD *)(v29 + 4) = 29810;
      *(_BYTE *)(v29 + 6) = 60;
      *(_QWORD *)(v8 + 32) += 7LL;
    }
    v31 = sub_CB59D0(v30, *(unsigned int *)(v9 + 96));
    v32 = *(_WORD **)(v31 + 32);
    if ( *(_QWORD *)(v31 + 24) - (_QWORD)v32 <= 1u )
    {
      sub_CB6200(v31, ">(", 2u);
    }
    else
    {
      *v32 = 10302;
      *(_QWORD *)(v31 + 32) += 2LL;
    }
  }
  v22 = *(_QWORD *)(v9 + 104);
  if ( (unsigned int)((*(_QWORD *)(v9 + 112) - v22) >> 3) )
  {
    v37 = (*(_QWORD *)(v9 + 112) - v22) >> 3;
    v23 = 0;
    while ( 1 )
    {
      v24 = v23++;
      (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v22 + 8 * v24) + 24LL))(
        *(_QWORD *)(v22 + 8 * v24),
        v8,
        v34,
        v7);
      if ( v37 <= v23 )
      {
        if ( v37 == v23 )
          break;
      }
      else
      {
        v25 = *(_BYTE **)(v8 + 32);
        if ( (unsigned __int64)v25 >= *(_QWORD *)(v8 + 24) )
        {
          sub_CB5D20(v8, 44);
        }
        else
        {
          *(_QWORD *)(v8 + 32) = v25 + 1;
          *v25 = 44;
        }
      }
      v22 = *(_QWORD *)(v9 + 104);
    }
  }
  if ( *(_DWORD *)(v9 + 96) )
  {
    v28 = *(_BYTE **)(v8 + 32);
    if ( (unsigned __int64)v28 >= *(_QWORD *)(v8 + 24) )
    {
      sub_CB5D20(v8, 41);
    }
    else
    {
      *(_QWORD *)(v8 + 32) = v28 + 1;
      *v28 = 41;
    }
  }
  result = *(_BYTE **)(v8 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v8 + 24) )
    return (_BYTE *)sub_CB5D20(v8, 41);
  *(_QWORD *)(v8 + 32) = result + 1;
  *result = 41;
  return result;
}
