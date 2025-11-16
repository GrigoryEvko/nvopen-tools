// Function: sub_31185E0
// Address: 0x31185e0
//
__int64 __fastcall sub_31185E0(__int64 a1, __int64 a2, unsigned int a3)
{
  void *v5; // rdi
  __int64 v6; // rdx
  _BYTE *v7; // r14
  size_t v8; // r13
  __int64 v9; // rax
  size_t v10[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a3 >= *(_DWORD *)(a2 + 40) )
  {
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  v5 = (void *)(a1 + 16);
  v6 = *(_QWORD *)(a2 + 32) + 32LL * a3;
  *(_QWORD *)a1 = v5;
  v7 = *(_BYTE **)v6;
  v8 = *(_QWORD *)(v6 + 8);
  if ( v8 + *(_QWORD *)v6 )
  {
    if ( !v7 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
  }
  v10[0] = *(_QWORD *)(v6 + 8);
  if ( v8 > 0xF )
  {
    v9 = sub_22409D0(a1, v10, 0);
    *(_QWORD *)a1 = v9;
    v5 = (void *)v9;
    *(_QWORD *)(a1 + 16) = v10[0];
    goto LABEL_12;
  }
  if ( v8 != 1 )
  {
    if ( !v8 )
      goto LABEL_8;
LABEL_12:
    memcpy(v5, v7, v8);
    v8 = v10[0];
    v5 = *(void **)a1;
    goto LABEL_8;
  }
  *(_BYTE *)(a1 + 16) = *v7;
LABEL_8:
  *(_QWORD *)(a1 + 8) = v8;
  *((_BYTE *)v5 + v8) = 0;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
