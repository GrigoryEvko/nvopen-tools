// Function: sub_DFCD70
// Address: 0xdfcd70
//
__int64 __fastcall sub_DFCD70(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 (__fastcall *v3)(); // rax
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // r12
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rbx
  unsigned __int8 **v10; // rcx
  int v11; // eax
  unsigned __int8 **v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // edx
  unsigned int v16; // r12d
  __int64 v18; // [rsp+0h] [rbp-70h]
  __int64 v19; // [rsp+8h] [rbp-68h]
  unsigned __int8 **v20; // [rsp+10h] [rbp-60h] BYREF
  __int64 v21; // [rsp+18h] [rbp-58h]
  _BYTE v22[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = *a1;
  v3 = *(__int64 (__fastcall **)())(*(_QWORD *)*a1 + 912LL);
  if ( v3 != sub_DFCF20 )
    return ((__int64 (__fastcall *)(_QWORD))v3)(*a1);
  v4 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(_QWORD *)(a2 - 8);
    v5 = v6 + v4;
  }
  else
  {
    v5 = a2;
    v6 = a2 - v4;
  }
  v7 = v5 - v6;
  v20 = (unsigned __int8 **)v22;
  v8 = v7 >> 5;
  v21 = 0x400000000LL;
  v9 = v7 >> 5;
  if ( (unsigned __int64)v7 > 0x80 )
  {
    v18 = v7;
    v19 = v7 >> 5;
    sub_C8D5F0((__int64)&v20, v22, v7 >> 5, 8u, v7, v8);
    v12 = v20;
    v11 = v21;
    LODWORD(v8) = v19;
    v7 = v18;
    v10 = &v20[(unsigned int)v21];
  }
  else
  {
    v10 = (unsigned __int8 **)v22;
    v11 = 0;
    v12 = (unsigned __int8 **)v22;
  }
  if ( v7 > 0 )
  {
    v13 = 0;
    do
    {
      v10[v13 / 8] = *(unsigned __int8 **)(v6 + 4 * v13);
      v13 += 8LL;
      --v9;
    }
    while ( v9 );
    v12 = v20;
    v11 = v21;
  }
  LODWORD(v21) = v8 + v11;
  v14 = sub_DFBE30((__int64 *)(v2 + 8), (unsigned __int8 *)a2, v12, (unsigned int)(v8 + v11), 3);
  if ( v15 )
    LODWORD(v6) = v15 >> 31;
  else
    LOBYTE(v6) = v14 <= 3;
  v16 = v6 ^ 1;
  if ( v20 != (unsigned __int8 **)v22 )
    _libc_free(v20, a2);
  return v16;
}
