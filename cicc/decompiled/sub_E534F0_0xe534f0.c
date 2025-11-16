// Function: sub_E534F0
// Address: 0xe534f0
//
__int64 __fastcall sub_E534F0(__int64 a1, char *a2)
{
  __int64 v2; // r12
  __int64 result; // rax
  _BYTE *v4; // rax
  void *v5; // rdx
  unsigned int v6; // r13d
  __int64 v7; // rax
  unsigned int v8; // ebx
  __int64 v9; // rax

  v2 = a1;
  result = *(unsigned int *)a2;
  if ( (_DWORD)result
    || (*((_DWORD *)a2 + 1) & 0x7FFFFFFF) != 0
    || (*((_DWORD *)a2 + 2) & 0x7FFFFFFF) != 0
    || (*((_DWORD *)a2 + 3) & 0x7FFFFFFF) != 0 )
  {
    v4 = *(_BYTE **)(a1 + 32);
    if ( (unsigned __int64)v4 >= *(_QWORD *)(a1 + 24) )
    {
      a1 = sub_CB5D20(a1, 9);
    }
    else
    {
      *(_QWORD *)(a1 + 32) = v4 + 1;
      *v4 = 9;
    }
    v5 = *(void **)(a1 + 32);
    if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 0xBu )
    {
      a1 = sub_CB6200(a1, "sdk_version ", 0xCu);
    }
    else
    {
      qmemcpy(v5, "sdk_version ", 12);
      *(_QWORD *)(a1 + 32) += 12LL;
    }
    result = sub_CB59D0(a1, *(unsigned int *)a2);
    if ( a2[7] < 0 )
    {
      v6 = *((_DWORD *)a2 + 1);
      v7 = sub_904010(v2, ", ");
      result = sub_CB59D0(v7, v6 & 0x7FFFFFFF);
      if ( a2[11] < 0 )
      {
        v8 = *((_DWORD *)a2 + 2);
        v9 = sub_904010(v2, ", ");
        return sub_CB59D0(v9, v8 & 0x7FFFFFFF);
      }
    }
  }
  return result;
}
