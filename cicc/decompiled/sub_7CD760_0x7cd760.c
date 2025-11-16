// Function: sub_7CD760
// Address: 0x7cd760
//
unsigned __int8 **__fastcall sub_7CD760(__int64 a1, int a2, __int64 *a3, __int64 a4)
{
  unsigned __int8 **result; // rax
  unsigned __int8 *v8; // rdi
  unsigned __int8 v9; // al
  __int64 v10; // rax
  _BOOL4 v11; // ecx
  __int64 v12; // rbx
  unsigned __int64 v13; // r15
  unsigned __int8 *v14; // rdi
  __int64 v15; // rax
  int v16; // [rsp+0h] [rbp-40h] BYREF
  _WORD v17[2]; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( !dword_4D0432C && !*(_BYTE *)(a1 + 43) )
    return sub_7CD070(a1, a2, a3, a4, 0, 0);
  v8 = **(unsigned __int8 ***)a1;
  v9 = *v8;
  if ( a2 )
  {
    if ( v9 == 92 )
      return sub_7CD070(a1, a2, a3, a4, 0, 0);
  }
  if ( !v9 )
    return sub_7CD070(a1, a2, a3, a4, 0, 0);
  if ( *(int *)(a1 + 16) > 0 )
    return sub_7CD070(a1, a2, a3, a4, 0, 0);
  v10 = *(_QWORD *)(a1 + 8);
  if ( v10 )
  {
    if ( *(unsigned __int8 **)(v10 + 8) == v8 )
      return sub_7CD070(a1, a2, a3, a4, 0, 0);
  }
  v11 = 0;
  if ( !*(_BYTE *)(a1 + 43) )
    v11 = unk_4F064A8 == 0;
  v12 = (int)sub_722680(v8, v18, &v16, v11);
  if ( v16 )
  {
    v14 = **(unsigned __int8 ***)a1;
    if ( *(_BYTE *)(a1 + 44) )
    {
      v13 = *v14;
      v12 = 1;
      v18[0] = v13;
    }
    else
    {
      v13 = 63;
      sub_7B0EB0((unsigned __int64)v14, (__int64)dword_4F07508);
      sub_684AC0(*(_BYTE *)(a1 + 42) == 0 ? 7 : 5, 0x366u);
      v18[0] = 63;
    }
  }
  else
  {
    v13 = v18[0];
  }
  if ( (v13 & ~a4) != 0 && *(_BYTE *)(a1 + 41) )
  {
    if ( (unsigned int)sub_722B20(v13, v17) == 2 )
    {
      v15 = v17[1];
      v13 = v17[0];
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 16) = 1;
      *(_QWORD *)(a1 + 32) = v15;
    }
    v18[0] = v13;
  }
  *a3 = v13;
  result = *(unsigned __int8 ***)a1;
  **(_QWORD **)a1 += v12;
  return result;
}
