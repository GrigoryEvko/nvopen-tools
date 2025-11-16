// Function: sub_27435A0
// Address: 0x27435a0
//
__int64 __fastcall sub_27435A0(__int64 a1, __int64 a2, __int64 a3)
{
  signed __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 result; // rax
  char v8; // r8
  char v9; // r8
  bool v10; // zf

  v4 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3);
  v5 = a1;
  if ( v4 >> 2 <= 0 )
  {
LABEL_11:
    if ( v4 != 2 )
    {
      if ( v4 != 3 )
      {
        if ( v4 != 1 )
          return a2;
LABEL_19:
        v10 = (unsigned __int8)sub_2743410(a3, *(_DWORD *)v5, *(_QWORD *)(v5 + 8), *(_BYTE **)(v5 + 16)) == 0;
        result = v5;
        if ( !v10 )
          return a2;
        return result;
      }
      v8 = sub_2743410(a3, *(_DWORD *)v5, *(_QWORD *)(v5 + 8), *(_BYTE **)(v5 + 16));
      result = v5;
      if ( !v8 )
        return result;
      v5 += 24;
    }
    v9 = sub_2743410(a3, *(_DWORD *)v5, *(_QWORD *)(v5 + 8), *(_BYTE **)(v5 + 16));
    result = v5;
    if ( !v9 )
      return result;
    v5 += 24;
    goto LABEL_19;
  }
  v6 = a1 + 96 * (v4 >> 2);
  while ( 1 )
  {
    if ( !(unsigned __int8)sub_2743410(a3, *(_DWORD *)v5, *(_QWORD *)(v5 + 8), *(_BYTE **)(v5 + 16)) )
      return v5;
    if ( !(unsigned __int8)sub_2743410(a3, *(_DWORD *)(v5 + 24), *(_QWORD *)(v5 + 32), *(_BYTE **)(v5 + 40)) )
      return v5 + 24;
    if ( !(unsigned __int8)sub_2743410(a3, *(_DWORD *)(v5 + 48), *(_QWORD *)(v5 + 56), *(_BYTE **)(v5 + 64)) )
      return v5 + 48;
    if ( !(unsigned __int8)sub_2743410(a3, *(_DWORD *)(v5 + 72), *(_QWORD *)(v5 + 80), *(_BYTE **)(v5 + 88)) )
      return v5 + 72;
    v5 += 96;
    if ( v6 == v5 )
    {
      v4 = 0xAAAAAAAAAAAAAAABLL * ((a2 - v5) >> 3);
      goto LABEL_11;
    }
  }
}
