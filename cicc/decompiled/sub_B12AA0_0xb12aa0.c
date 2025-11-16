// Function: sub_B12AA0
// Address: 0xb12aa0
//
__int64 __fastcall sub_B12AA0(__int64 a1, __int64 a2, char *a3, __int64 a4)
{
  char v6; // al
  _BYTE *v7; // rdx
  __int64 v8; // r13
  __int64 result; // rax
  unsigned __int8 *v10; // r13
  unsigned int i; // r15d
  unsigned __int8 *v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 *v18; // r13
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // r12
  unsigned __int8 *v24; // [rsp+8h] [rbp-68h]
  __int64 *v25; // [rsp+10h] [rbp-60h] BYREF
  __int64 v26; // [rsp+18h] [rbp-58h]
  _BYTE v27[80]; // [rsp+20h] [rbp-50h] BYREF

  v6 = *a3;
  v7 = *(_BYTE **)(a1 + 40);
  if ( *v7 != 4 )
  {
    if ( v6 == 24 )
      v8 = *((_QWORD *)a3 + 3);
    else
      v8 = sub_B98A20(a3, a2, v7, a4);
    sub_B91340(a1 + 40, 0);
    *(_QWORD *)(a1 + 40) = v8;
    return sub_B96F50(a1 + 40, 0);
  }
  v25 = (__int64 *)v27;
  v26 = 0x400000000LL;
  if ( v6 == 24 )
  {
    v10 = (unsigned __int8 *)*((_QWORD *)a3 + 3);
    if ( (unsigned int)*v10 - 1 >= 2 )
      v10 = 0;
  }
  else
  {
    v10 = (unsigned __int8 *)sub_B98A20(a3, 0x400000000LL, v7, a4);
  }
  for ( i = 0; (unsigned int)sub_B12A30(a1) > i; ++i )
  {
    v12 = v10;
    if ( i != (_DWORD)a2 )
    {
      v15 = sub_B12A50(a1, i);
      if ( *(_BYTE *)v15 != 24 )
      {
        v12 = (unsigned __int8 *)sub_B98A20(v15, i, v16, v17);
        v13 = (unsigned int)v26;
        v14 = (unsigned int)v26 + 1LL;
        if ( v14 <= HIDWORD(v26) )
          goto LABEL_13;
        goto LABEL_18;
      }
      v12 = *(unsigned __int8 **)(v15 + 24);
      if ( (unsigned int)*v12 - 1 > 1 )
        v12 = 0;
    }
    v13 = (unsigned int)v26;
    v14 = (unsigned int)v26 + 1LL;
    if ( v14 <= HIDWORD(v26) )
      goto LABEL_13;
LABEL_18:
    v24 = v12;
    sub_C8D5F0(&v25, v27, v14, 8);
    v13 = (unsigned int)v26;
    v12 = v24;
LABEL_13:
    v25[v13] = (__int64)v12;
    LODWORD(v26) = v26 + 1;
  }
  v18 = v25;
  v19 = (unsigned int)v26;
  v20 = sub_B12A50(a1, 0);
  v22 = (__int64 *)sub_BD5C60(v20, 0, v21);
  v23 = sub_B00B60(v22, v18, v19);
  sub_B91340(a1 + 40, 0);
  *(_QWORD *)(a1 + 40) = v23;
  result = sub_B96F50(a1 + 40, 0);
  if ( v25 != (__int64 *)v27 )
    return _libc_free(v25, 0);
  return result;
}
