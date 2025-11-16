// Function: sub_85F1C0
// Address: 0x85f1c0
//
__int64 __fastcall sub_85F1C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, unsigned int a7)
{
  int v7; // eax
  bool v10; // zf
  __int64 v11; // rdx
  int v12; // r12d
  __int64 result; // rax
  __int64 v14; // rbx
  __int64 i; // r12

  v10 = a3 == 0;
  v11 = a4;
  LOBYTE(v7) = !v10;
  v12 = a7 & 0x200000;
  LOBYTE(a4) = (a7 & 0x200000) == 0;
  result = (unsigned int)a4 & v7;
  if ( a2 )
  {
    if ( (_BYTE)result )
    {
      v12 = 0;
LABEL_4:
      result = sub_85F2F0(a1, a3, v11, a5, a6);
LABEL_5:
      if ( v12 )
      {
        v14 = *(_QWORD *)(a2 + 184);
        result = (__int64)qword_4F04C68;
        for ( i = qword_4F04C68[0] + 776LL * dword_4F04C64; v14; v14 = *(_QWORD *)v14 )
          result = sub_85EE10(v14, i, *(_DWORD *)(v14 + 56));
      }
      return result;
    }
    result = *(unsigned __int8 *)(a2 + 28);
    if ( (_BYTE)result == 2 )
    {
      if ( !v12 )
        return result;
    }
    else
    {
      if ( (((_BYTE)result - 15) & 0xFD) == 0 && !v12 )
        return result;
      if ( (_BYTE)result == 6 )
      {
        if ( !a3 )
          goto LABEL_5;
        goto LABEL_4;
      }
      if ( (unsigned __int8)result <= 6u )
        goto LABEL_5;
      if ( (_BYTE)result != 15 )
      {
        if ( (_BYTE)result == 17 )
          result = sub_85F680(a1, a2, v11, *(_QWORD *)(a2 + 32), a7);
        goto LABEL_5;
      }
    }
    result = sub_85F680(a1, a2, v11, 0, a7);
    goto LABEL_5;
  }
  if ( (_BYTE)result )
    return sub_85F2F0(a1, a3, v11, a5, a6);
  return result;
}
