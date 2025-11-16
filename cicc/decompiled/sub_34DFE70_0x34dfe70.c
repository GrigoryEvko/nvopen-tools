// Function: sub_34DFE70
// Address: 0x34dfe70
//
__int64 __fastcall sub_34DFE70(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        int a5,
        unsigned __int16 ***a6,
        __int64 a7)
{
  __int64 v9; // rdi
  __int64 v10; // rbx
  unsigned __int16 *v11; // r8
  __int64 *v12; // r14
  unsigned __int16 *v13; // r12
  unsigned int v14; // ebx
  __int64 v15; // rax
  unsigned int *v16; // r13
  __int64 v17; // r12
  unsigned int v18; // edx
  unsigned __int16 *v20; // [rsp+8h] [rbp-58h]
  unsigned __int16 *v21; // [rsp+20h] [rbp-40h]

  v9 = a1[5];
  v10 = *(_QWORD *)v9 + 24LL * *((unsigned __int16 *)*a6 + 12);
  if ( *(_DWORD *)(v9 + 8) != *(_DWORD *)v10 )
    sub_2F60630(v9, a6);
  v11 = *(unsigned __int16 **)(v10 + 16);
  v21 = &v11[*(unsigned int *)(v10 + 4)];
  if ( v21 == v11 )
    return 0;
  v12 = a1;
  v13 = *(unsigned __int16 **)(v10 + 16);
  while ( 1 )
  {
    while ( 1 )
    {
      v14 = *v13;
      if ( v14 != a4 && v14 != a5 && !sub_34DFD80((__int64)v12, a2, a3, *v13) )
      {
        v15 = v12[24];
        if ( *(_DWORD *)(v15 + 4LL * (unsigned __int16)v14) == -1
          && *(_QWORD *)(v12[15] + 8LL * (unsigned __int16)v14) != -1
          && *(_DWORD *)(v15 + 4LL * a4) <= *(_DWORD *)(v12[27] + 4LL * (unsigned __int16)v14) )
        {
          break;
        }
      }
      if ( v21 == ++v13 )
        return 0;
    }
    if ( *(_QWORD *)a7 + 4LL * *(unsigned int *)(a7 + 8) == *(_QWORD *)a7 )
      return v14;
    v20 = v13;
    v16 = *(unsigned int **)a7;
    v17 = *(_QWORD *)a7 + 4LL * *(unsigned int *)(a7 + 8);
    while ( 1 )
    {
      v18 = *v16;
      if ( v14 == *v16 )
        break;
      if ( v14 - 1 <= 0x3FFFFFFE && v18 - 1 <= 0x3FFFFFFE )
      {
        if ( (unsigned __int8)sub_E92070(v12[4], v14, v18) )
          break;
        if ( (unsigned int *)v17 == ++v16 )
          return v14;
      }
      else if ( (unsigned int *)v17 == ++v16 )
      {
        return v14;
      }
    }
    v13 = v20 + 1;
    if ( v21 == v20 + 1 )
      return 0;
  }
}
