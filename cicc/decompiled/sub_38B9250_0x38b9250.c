// Function: sub_38B9250
// Address: 0x38b9250
//
__int64 __fastcall sub_38B9250(__int64 a1)
{
  __int64 v1; // rcx
  __int64 result; // rax
  unsigned __int64 *v4; // rdi
  unsigned __int64 *v5; // rdx
  unsigned __int64 *v6; // r14
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // r12
  unsigned __int64 *v9; // r15
  __int64 v10; // r13
  unsigned __int64 *v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rdx
  _QWORD *v14; // r12
  _QWORD *v15; // r15
  void (__fastcall *v16)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v17; // rax
  _QWORD *v18; // [rsp-40h] [rbp-40h]

  v1 = *(unsigned int *)(a1 + 340);
  result = 0;
  if ( *(_DWORD *)(a1 + 344) == (_DWORD)v1 )
    return result;
  v4 = *(unsigned __int64 **)(a1 + 328);
  v5 = *(unsigned __int64 **)(a1 + 320);
  v6 = &v4[v1];
  if ( v4 != v5 )
    v6 = &v4[*(unsigned int *)(a1 + 336)];
  if ( v4 == v6 )
  {
LABEL_8:
    v10 = a1 + 312;
  }
  else
  {
    v7 = v4;
    while ( 1 )
    {
      v8 = *v7;
      v9 = v7;
      if ( *v7 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v6 == ++v7 )
        goto LABEL_8;
    }
    v10 = a1 + 312;
    if ( v6 != v7 )
    {
      do
      {
        sub_157FA10(v8);
        sub_38B8D50(a1, v8);
        if ( v8 )
        {
          sub_157EF40(v8);
          j_j___libc_free_0(v8);
        }
        v11 = v9 + 1;
        if ( v9 + 1 == v6 )
          break;
        while ( 1 )
        {
          v8 = *v11;
          v9 = v11;
          if ( *v11 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v6 == ++v11 )
            goto LABEL_15;
        }
      }
      while ( v6 != v11 );
LABEL_15:
      v4 = *(unsigned __int64 **)(a1 + 328);
      v5 = *(unsigned __int64 **)(a1 + 320);
    }
  }
  ++*(_QWORD *)(a1 + 312);
  if ( v5 == v4 )
    goto LABEL_21;
  v12 = 4 * (*(_DWORD *)(a1 + 340) - *(_DWORD *)(a1 + 344));
  v13 = *(unsigned int *)(a1 + 336);
  if ( v12 < 0x20 )
    v12 = 32;
  if ( (unsigned int)v13 <= v12 )
  {
    memset(v4, -1, 8 * v13);
LABEL_21:
    *(_QWORD *)(a1 + 340) = 0;
    goto LABEL_22;
  }
  sub_16CC920(v10);
LABEL_22:
  v14 = *(_QWORD **)(a1 + 424);
  v18 = *(_QWORD **)(a1 + 416);
  result = 1;
  if ( v18 != v14 )
  {
    v15 = *(_QWORD **)(a1 + 416);
    do
    {
      v16 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v15[7];
      *v15 = &unk_49F0628;
      if ( v16 )
        v16(v15 + 5, v15 + 5, 3);
      *v15 = &unk_49EE2B0;
      v17 = v15[3];
      if ( v17 != -8 && v17 != 0 && v17 != -16 )
        sub_1649B30(v15 + 1);
      v15 += 9;
    }
    while ( v14 != v15 );
    *(_QWORD *)(a1 + 424) = v18;
    return 1;
  }
  return result;
}
