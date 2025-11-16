// Function: sub_1E85000
// Address: 0x1e85000
//
__int64 __fastcall sub_1E85000(_QWORD **a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rcx
  bool v4; // dl
  unsigned __int8 v5; // si
  unsigned int v6; // r8d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  _BYTE *v12; // rax

  if ( *(_BYTE *)a2 )
    return 1;
  v2 = *(_DWORD *)(a2 + 8);
  if ( v2 >= 0 || (v10 = v2 & 0x7FFFFFFF, v11 = (*a1)[6], (unsigned int)v10 >= *(_DWORD *)(v11 + 336)) )
  {
    v3 = 0;
    v4 = 0;
    v5 = 0;
  }
  else
  {
    v12 = (_BYTE *)(*(_QWORD *)(v11 + 328) + 8 * v10);
    v5 = *v12 & 1;
    v4 = (*v12 & 2) != 0;
    v3 = *(_QWORD *)v12 >> 2;
  }
  v6 = 0;
  v7 = (4 * v3) | v5 | (2LL * v4);
  if ( (4 * v3) | (unsigned __int16)(v5 | (unsigned __int16)(2 * v4)) & 0xFFFC )
  {
    v8 = *a1[1];
    if ( (((unsigned __int8)v8 ^ (unsigned __int8)v7) & 3) == 0 && ((v8 ^ v7) & 0xFFFFFFFFFFFFFFFCLL) == 0 )
      return 1;
  }
  return v6;
}
