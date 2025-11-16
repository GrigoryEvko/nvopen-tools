// Function: sub_2A893F0
// Address: 0x2a893f0
//
__int64 __fastcall sub_2A893F0(__int64 a1, unsigned int a2, __int64 **a3, _BYTE *a4)
{
  __int64 *v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int8 *v13; // rdi
  unsigned __int8 *v14; // r15
  __int64 result; // rax
  __int64 v16; // r14
  __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  __int64 v19; // [rsp+18h] [rbp-38h]

  v7 = *a3;
  v8 = sub_9208B0((__int64)a4, (__int64)a3);
  v10 = v9;
  v18 = v8;
  v11 = v8;
  v12 = *(_QWORD *)(a1 - 32);
  v19 = v10;
  if ( !v12 || *(_BYTE *)v12 || *(_QWORD *)(v12 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v13 = *(unsigned __int8 **)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  if ( ((*(_DWORD *)(v12 + 36) - 243) & 0xFFFFFFFD) != 0 )
  {
    v14 = sub_BD3990(v13, v11);
    LODWORD(v19) = sub_AE43F0((__int64)a4, *((_QWORD *)v14 + 1));
    if ( (unsigned int)v19 > 0x40 )
      sub_C43690((__int64)&v18, a2, 0);
    else
      v18 = a2;
    result = sub_971820((__int64)v14, (__int64)a3, (__int64)&v18, a4);
    if ( (unsigned int)v19 > 0x40 )
    {
      if ( v18 )
      {
        v17 = result;
        j_j___libc_free_0_0(v18);
        return v17;
      }
    }
  }
  else if ( *v13 == 17 )
  {
    sub_C47700((__int64)&v18, 8 * (v11 >> 3), (__int64)(v13 + 24));
    v16 = sub_ACCFD0(v7, (__int64)&v18);
    if ( (unsigned int)v19 > 0x40 && v18 )
      j_j___libc_free_0_0(v18);
    return sub_9717D0(v16, (__int64)a3, a4);
  }
  else
  {
    return 0;
  }
  return result;
}
