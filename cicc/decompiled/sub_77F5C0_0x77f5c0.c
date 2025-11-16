// Function: sub_77F5C0
// Address: 0x77f5c0
//
__int64 __fastcall sub_77F5C0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned int *a4,
        _DWORD *a5,
        unsigned int *a6,
        _DWORD *a7)
{
  char v8; // al
  unsigned int v9; // ecx
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // eax
  unsigned int *v14; // [rsp+0h] [rbp-30h]
  _DWORD *v15; // [rsp+8h] [rbp-28h]

  v8 = *(_BYTE *)(a3 + 140);
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
  {
    if ( (unsigned __int8)(v8 - 2) > 1u )
    {
      v14 = a6;
      v15 = a5;
      v13 = sub_7764B0(a1, a3, a7);
      a6 = v14;
      a5 = v15;
      *v14 = v13;
      result = (unsigned int)*a7;
      if ( !(_DWORD)result )
        goto LABEL_7;
    }
    else
    {
      *a6 = 16;
      result = (unsigned int)*a7;
      if ( !(_DWORD)result )
      {
LABEL_7:
        *a4 = 0;
        goto LABEL_8;
      }
    }
    if ( (*(_BYTE *)(a2 + 8) & 8) == 0 )
    {
      *a4 = 1;
      result = (*(unsigned __int8 *)(a2 + 8) >> 1) & 1;
      *a5 = result;
      return result;
    }
    *a4 = *(_DWORD *)(a2 + 8) >> 8;
    v11 = *(_QWORD *)(a2 + 16);
    if ( (*(_BYTE *)(a2 + 8) & 4) != 0 )
      v11 = *(_QWORD *)(v11 + 24);
    v12 = *(_QWORD *)a2 - v11;
    *a5 = v12;
    result = (unsigned int)v12;
    if ( *a6 )
    {
      result = (unsigned int)v12 / *a6;
      *a5 = result;
      return result;
    }
LABEL_8:
    *a5 = 0;
    return result;
  }
  v9 = 1;
  if ( v8 != 1 )
    v9 = *(_DWORD *)(a3 + 128);
  *a6 = v9;
  return sub_771560(a1, *(_QWORD *)(a2 + 16), a3, v9, a4, a5, a7);
}
