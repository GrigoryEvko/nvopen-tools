// Function: sub_2CB0FF0
// Address: 0x2cb0ff0
//
__int64 __fastcall sub_2CB0FF0(
        unsigned __int64 a1,
        unsigned __int8 *a2,
        int *a3,
        __int64 a4,
        int *a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // rax
  unsigned int v11; // eax
  __int64 v12; // rcx
  unsigned int v13; // r8d
  unsigned int v14; // eax
  __int64 result; // rax
  __int64 v16; // rcx
  unsigned int v17; // r8d
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // [rsp-8h] [rbp-58h]
  int v23; // [rsp+8h] [rbp-48h]
  int v24; // [rsp+Ch] [rbp-44h]
  unsigned __int8 v25; // [rsp+Ch] [rbp-44h]
  unsigned __int8 v26; // [rsp+Ch] [rbp-44h]
  int v27[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v7 = *(_QWORD *)(a4 + 8);
  if ( *(_BYTE *)(v7 + 8) != 12 )
    return 0;
  v23 = *a5;
  v24 = *(_DWORD *)(v7 + 8) >> 8;
  v11 = sub_9AF8B0(a4, a1, 0, 0, 0, 0, 1);
  v12 = v21;
  v13 = v11;
  v14 = v24 - v11;
  if ( v14 > 7 )
  {
    if ( v14 != 8 )
      return 0;
    if ( v23 )
    {
      if ( v23 != 2 )
        return 0;
    }
    else
    {
      v27[0] = 1;
      v13 = sub_2CAFEF0(a1, (unsigned __int8 *)a4, v27, v21, v13);
      result = 0;
      if ( (_BYTE)v13 )
        return result;
    }
  }
  if ( !(unsigned __int8)sub_2CAFEF0(a1, a2, a3, v12, v13) )
    return 0;
  result = sub_2CAFEF0(a1, (unsigned __int8 *)a4, a5, v16, v17);
  if ( !(_BYTE)result )
    return 0;
  v19 = *(unsigned int *)(a6 + 8);
  if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
  {
    v25 = result;
    sub_C8D5F0(a6, (const void *)(a6 + 16), v19 + 1, 8u, v19 + 1, v18);
    v19 = *(unsigned int *)(a6 + 8);
    result = v25;
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v19) = a2;
  ++*(_DWORD *)(a6 + 8);
  v20 = *(unsigned int *)(a7 + 8);
  if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a7 + 12) )
  {
    v26 = result;
    sub_C8D5F0(a7, (const void *)(a7 + 16), v20 + 1, 8u, v20 + 1, v18);
    v20 = *(unsigned int *)(a7 + 8);
    result = v26;
  }
  *(_QWORD *)(*(_QWORD *)a7 + 8 * v20) = a4;
  ++*(_DWORD *)(a7 + 8);
  return result;
}
