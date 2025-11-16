// Function: sub_2436CE0
// Address: 0x2436ce0
//
__int64 __fastcall sub_2436CE0(__int64 a1, __int64 *a2, unsigned __int64 a3)
{
  __int64 v3; // rax
  int v5; // eax
  __int64 *v7; // r15
  unsigned int v8; // r13d
  __int64 v10; // rax
  unsigned __int8 *v11; // rax
  __int64 *v12; // rdi
  unsigned __int8 *v13; // rax
  __int64 v14[6]; // [rsp-30h] [rbp-30h] BYREF

  v3 = *(_QWORD *)(a3 - 32);
  if ( !v3 )
    BUG();
  if ( *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a3 + 80) )
    BUG();
  v5 = *(_DWORD *)(v3 + 36);
  v7 = a2;
  if ( v5 == 238 || (unsigned int)(v5 - 240) <= 1 )
  {
    if ( byte_4FE5528 )
    {
      v13 = sub_BD3990(*(unsigned __int8 **)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)), (__int64)a2);
      v14[0] = a3;
      a2 = v14;
      v12 = v7;
      v8 = sub_2433C90(a1, a3, (__int64)v13);
      if ( !(_BYTE)v8 )
        goto LABEL_15;
      sub_2434B50(v7, v14);
    }
    if ( !byte_4FE5608 )
      return 1;
    v10 = 1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
LABEL_13:
    v11 = sub_BD3990(*(unsigned __int8 **)(a3 + 32 * v10), (__int64)a2);
    v14[0] = a3;
    v12 = v7;
    v8 = sub_2433C90(a1, a3, (__int64)v11);
    if ( (_BYTE)v8 )
    {
      sub_2434B50(v7, v14);
      return v8;
    }
LABEL_15:
    sub_2434C80(v12, v14);
    return v8;
  }
  v8 = 0;
  if ( ((v5 - 243) & 0xFFFFFFFD) != 0 )
    return v8;
  if ( byte_4FE5528 )
  {
    v10 = -(__int64)(*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
    goto LABEL_13;
  }
  return 1;
}
