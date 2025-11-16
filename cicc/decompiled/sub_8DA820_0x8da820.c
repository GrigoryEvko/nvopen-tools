// Function: sub_8DA820
// Address: 0x8da820
//
__int64 __fastcall sub_8DA820(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, unsigned int a6, int a7)
{
  __int64 result; // rax
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // r10
  __int64 v11; // r8
  __int64 v12; // rbx
  __int64 v13; // r11
  __int64 v14; // r10

  result = 1;
  if ( a1 == a2 )
    return result;
  result = dword_4F07588;
  if ( a1 && a2 )
  {
    if ( !dword_4F07588 )
      return result;
    v8 = *(_QWORD *)(a1 + 32);
    if ( *(_QWORD *)(a2 + 32) == v8 )
    {
      result = 1;
      if ( v8 )
        return result;
    }
  }
  else if ( !dword_4F07588 )
  {
    return result;
  }
  if ( (*(_BYTE *)(a1 + 177) & 0x20) != 0 && (*(_BYTE *)(a2 + 177) & 0x20) != 0 )
  {
    v9 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a2;
    if ( *(_QWORD *)a1 )
      goto LABEL_10;
    return 0;
  }
  result = 0;
  if ( !a3 )
    return result;
  v9 = *(_QWORD *)a1;
  v10 = *(_QWORD *)a2;
  if ( !*(_QWORD *)a1 )
    return 0;
LABEL_10:
  if ( !v10 )
    return 0;
  v11 = (unsigned __int8)((_DWORD)a4 != 0) << 6;
  if ( a6 )
    v11 = (((_DWORD)a4 != 0) << 6) | 0x100u;
  v12 = *(_QWORD *)(a1 + 168);
  if ( a7 )
    v11 = (unsigned int)v11 | 0x4010;
  if ( *(_QWORD *)(v12 + 256) && *(_QWORD *)(*(_QWORD *)(a2 + 168) + 256LL) )
    return (unsigned int)sub_8D97D0(*(_QWORD *)(v12 + 256), *(_QWORD *)(*(_QWORD *)(a2 + 168) + 256LL), v11, a4, v11) != 0;
  v13 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 72LL);
  result = 0;
  if ( v13 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(v10 + 96) + 72LL);
    if ( v14 )
      return sub_89BAF0(
               v13,
               v14,
               a1,
               a2,
               *(_QWORD *)(v12 + 168),
               *(_QWORD *)(*(_QWORD *)(a2 + 168) + 168LL),
               0,
               a3,
               a4,
               a6,
               a7);
  }
  return result;
}
