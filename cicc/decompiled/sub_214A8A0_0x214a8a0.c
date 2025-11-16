// Function: sub_214A8A0
// Address: 0x214a8a0
//
__int64 __fastcall sub_214A8A0(__int64 a1, _BYTE *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // eax
  __int64 v9; // r12
  __int64 v10; // rdx
  size_t v11; // rax
  _BYTE *v12; // rdi
  __int64 result; // rax
  __int64 v14; // rax
  size_t v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v8 = *(_DWORD *)(a1 + 24);
  if ( v8 >= *(_DWORD *)(a1 + 28) )
  {
    sub_12BE710(a1 + 16, 0, a3, a4, a5, a6);
    v8 = *(_DWORD *)(a1 + 24);
  }
  v9 = *(_QWORD *)(a1 + 16) + 32LL * v8;
  if ( v9 )
  {
    v10 = v9 + 16;
    if ( !a2 )
    {
      *(_QWORD *)v9 = v10;
      *(_QWORD *)(v9 + 8) = 0;
      *(_BYTE *)(v9 + 16) = 0;
      v8 = *(_DWORD *)(a1 + 24);
      goto LABEL_9;
    }
    *(_QWORD *)v9 = v10;
    v11 = a3;
    v15[0] = a3;
    if ( a3 > 0xF )
    {
      v14 = sub_22409D0(v9, v15, 0);
      *(_QWORD *)v9 = v14;
      v12 = (_BYTE *)v14;
      *(_QWORD *)(v9 + 16) = v15[0];
    }
    else
    {
      v12 = *(_BYTE **)v9;
      if ( a3 == 1 )
      {
        *v12 = *a2;
        v11 = v15[0];
        v12 = *(_BYTE **)v9;
LABEL_8:
        *(_QWORD *)(v9 + 8) = v11;
        v12[v11] = 0;
        v8 = *(_DWORD *)(a1 + 24);
        goto LABEL_9;
      }
      if ( !a3 )
        goto LABEL_8;
    }
    memcpy(v12, a2, a3);
    v11 = v15[0];
    v12 = *(_BYTE **)v9;
    goto LABEL_8;
  }
LABEL_9:
  result = v8 + 1;
  *(_DWORD *)(a1 + 24) = result;
  return result;
}
