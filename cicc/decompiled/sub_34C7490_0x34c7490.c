// Function: sub_34C7490
// Address: 0x34c7490
//
__int64 __fastcall sub_34C7490(__int64 a1, int a2, _QWORD *a3, __int64 a4)
{
  _DWORD *v8; // rcx
  unsigned int v9; // esi
  int v10; // edx
  int v11; // r12d
  __int64 *v12; // rbx
  __int64 v13; // rax
  int v14; // edx
  __int64 result; // rax

  v8 = *(_DWORD **)(a1 + 32);
  v9 = v8[2];
  v10 = (*v8 >> 8) & 0xFFF;
  v11 = (v8[10] >> 8) & 0xFFF;
  if ( v9 == a2 )
  {
    v9 = v8[12];
    v11 = (*v8 >> 8) & 0xFFF;
    v10 = (v8[10] >> 8) & 0xFFF;
  }
  if ( !v9 )
    return 0;
  if ( (v9 & 0x80000000) != 0 )
  {
    result = 0;
    if ( v11 == v10 )
      return v9;
  }
  else
  {
    v12 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a4 + 56) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v10 )
      v9 = sub_E91CF0(a3, v9, v10);
    if ( v9 - 1 <= 0x3FFFFFFE )
    {
      v13 = v9 >> 3;
      if ( (unsigned int)v13 < *(unsigned __int16 *)(*v12 + 22) )
      {
        v14 = *(unsigned __int8 *)(*(_QWORD *)(*v12 + 8) + v13);
        if ( _bittest(&v14, v9 & 7) )
          return v9;
      }
    }
    if ( !v11 )
      return 0;
    return sub_E91D60(a3, v9, v11, *v12);
  }
  return result;
}
