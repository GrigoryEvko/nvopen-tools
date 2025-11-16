// Function: sub_15369D0
// Address: 0x15369d0
//
__int64 __fastcall sub_15369D0(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r9
  _BYTE *v8; // r8
  size_t v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 v17; // rax
  void *v18; // rdi
  __int64 v19; // rdi
  _BYTE *src; // [rsp+0h] [rbp-50h]
  __int64 v21; // [rsp+8h] [rbp-48h]
  size_t v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = sub_22077B0(72);
  v6 = *a3;
  v7 = v5 + 32;
  *(_QWORD *)(v5 + 32) = v5 + 48;
  v8 = *(_BYTE **)v6;
  v9 = *(_QWORD *)(v6 + 8);
  if ( v9 + *(_QWORD *)v6 && !v8 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v22[0] = *(_QWORD *)(v6 + 8);
  if ( v9 > 0xF )
  {
    src = v8;
    v17 = sub_22409D0(v5 + 32, v22, 0);
    v7 = v5 + 32;
    v8 = src;
    *(_QWORD *)(v5 + 32) = v17;
    v18 = (void *)v17;
    *(_QWORD *)(v5 + 48) = v22[0];
  }
  else
  {
    if ( v9 == 1 )
    {
      *(_BYTE *)(v5 + 48) = *v8;
      v10 = v5 + 48;
      goto LABEL_6;
    }
    if ( !v9 )
    {
      v10 = v5 + 48;
      goto LABEL_6;
    }
    v18 = (void *)(v5 + 48);
  }
  v21 = v7;
  memcpy(v18, v8, v9);
  v9 = v22[0];
  v10 = *(_QWORD *)(v5 + 32);
  v7 = v21;
LABEL_6:
  *(_QWORD *)(v5 + 40) = v9;
  *(_BYTE *)(v10 + v9) = 0;
  *(_DWORD *)(v5 + 64) = 0;
  v11 = sub_A288A0(a1, a2, v7);
  v13 = v11;
  v14 = v12;
  if ( v12 )
  {
    v15 = 1;
    if ( !v11 && (_QWORD *)v12 != a1 + 1 )
      v15 = (unsigned int)sub_15238C0(
                            *(const void **)(v5 + 32),
                            *(_QWORD *)(v5 + 40),
                            *(const void **)(v12 + 32),
                            *(_QWORD *)(v12 + 40)) >> 31;
    sub_220F040(v15, v5, v14, a1 + 1);
    ++a1[5];
    return v5;
  }
  else
  {
    v19 = *(_QWORD *)(v5 + 32);
    if ( v5 + 48 != v19 )
      j_j___libc_free_0(v19, *(_QWORD *)(v5 + 48) + 1LL);
    j_j___libc_free_0(v5, 72);
    return v13;
  }
}
