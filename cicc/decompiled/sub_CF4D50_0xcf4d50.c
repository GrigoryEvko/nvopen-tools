// Function: sub_CF4D50
// Address: 0xcf4d50
//
__int64 __fastcall sub_CF4D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v7; // eax
  _QWORD *v8; // r15
  int v11; // eax
  unsigned __int8 v12; // di
  int v13; // esi
  int v14; // edx
  _QWORD *v16; // [rsp+8h] [rbp-38h]

  v7 = *(_DWORD *)(a4 + 352);
  *(_DWORD *)(a4 + 352) = v7 + 1;
  v8 = *(_QWORD **)(a1 + 8);
  v16 = *(_QWORD **)(a1 + 16);
  if ( v16 == v8 )
  {
    v13 = 0;
    LOBYTE(v14) = 0;
    v12 = 1;
  }
  else
  {
    do
    {
      v11 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(*(_QWORD *)*v8 + 16LL))(
              *v8,
              a2,
              a3,
              a4,
              a5);
      v12 = v11;
      v13 = v11 >> 9;
      v14 = ((unsigned int)v11 >> 8) & 1;
      if ( (_BYTE)v11 != 1 )
        break;
      ++v8;
    }
    while ( v16 != v8 );
    v7 = *(_DWORD *)(a4 + 352) - 1;
  }
  *(_DWORD *)(a4 + 352) = v7;
  return (v13 << 9) | ((unsigned __int8)v14 << 8) | (unsigned int)v12;
}
