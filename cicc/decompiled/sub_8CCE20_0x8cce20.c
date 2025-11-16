// Function: sub_8CCE20
// Address: 0x8cce20
//
__int64 __fastcall sub_8CCE20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 **v10; // rdi
  __int64 *v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // kr00_8
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // r14
  _QWORD *v19; // rax

  result = (__int64)&qword_4F074A0;
  if ( qword_4F074B0 != qword_4F60258 )
    return result;
  v6 = a2;
  v8 = *(_QWORD *)(a2 + 104);
  result = *(unsigned __int8 *)(a1 + 80);
  if ( !v8 )
  {
    if ( (unsigned __int8)(result - 10) <= 1u || (_BYTE)result == 17 )
    {
      result = *(_QWORD *)(a1 + 96);
      if ( result )
      {
        if ( (*(_BYTE *)(result + 80) & 4) != 0 )
        {
          if ( !dword_4D03FE8[0] || *qword_4D03FD0 )
          {
            result = (__int64)sub_878440();
            v14 = qword_4F60240;
            *(_QWORD *)(result + 8) = a1;
            *(_QWORD *)result = v14;
            qword_4F60240 = result;
            return result;
          }
          return sub_8CA1D0(v6, a1);
        }
      }
    }
    return result;
  }
  v9 = (unsigned int)(result - 4);
  if ( (unsigned __int8)(result - 4) > 2u )
  {
    if ( (_BYTE)result != 3 )
      goto LABEL_8;
    a5 = *(_QWORD *)(a1 + 88);
    if ( !*(_BYTE *)(a1 + 104) )
    {
      if ( (*(_BYTE *)(a5 + 186) & 0x50) != 0x10 )
        goto LABEL_8;
      return (__int64)sub_8C7090(6, a5);
    }
  }
  else
  {
    a5 = *(_QWORD *)(a1 + 88);
  }
  if ( (*(_BYTE *)(a5 + 177) & 0xA0) == 0x20 )
    return (__int64)sub_8C7090(6, a5);
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0
    && *(char *)(a5 + 177) < 0
    && (*(_BYTE *)(*(_QWORD *)(a1 + 64) + 177LL) & 0x30) == 0x10 )
  {
    return result;
  }
LABEL_8:
  if ( dword_4D03FE8[0] )
  {
    if ( !*qword_4D03FD0 )
      return sub_8CA1D0(v6, a1);
    v10 = *(__int64 ***)(v8 + 32);
    v11 = *(__int64 **)(a2 + 104);
    if ( v10 )
      v11 = *v10;
    if ( *(__int64 **)(v8 + 200) != v11 )
    {
      v12 = *v11;
      LODWORD(v8) = *(unsigned __int8 *)(v12 + 80) - 4;
      v13 = (unsigned int)v8;
      v8 = (unsigned __int8)v8;
      switch ( (char)v8 )
      {
        case 0:
        case 1:
          v8 = *(_QWORD *)(v12 + 96);
          v6 = *(_QWORD *)(v8 + 80);
          break;
        case 2:
          v8 = *(_QWORD *)(v12 + 96);
          v6 = *(_QWORD *)(v8 + 32);
          break;
        case 5:
        case 6:
          v8 = *(_QWORD *)(v12 + 96);
          v6 = *(_QWORD *)(v8 + 56);
          break;
        case 15:
        case 16:
        case 17:
        case 18:
          v6 = *(_QWORD *)(v12 + 88);
          break;
        default:
          v8 = v13;
          v6 = 0;
          break;
      }
    }
    if ( (unsigned __int8)(result - 4) > 1u )
    {
      if ( (unsigned __int8)(result - 10) > 1u && (_BYTE)result != 17 )
        return result;
      v15 = sub_8C6FA0(
              *(_QWORD **)(v6 + 112),
              *(_QWORD *)(a1 + 88),
              (unsigned int)(result - 10),
              v9,
              (_UNKNOWN *__ptr32 *)a5);
      if ( v15 )
      {
        v16 = *(_QWORD *)(a1 + 88);
        v17 = *(_QWORD *)(v15[1] + 88LL);
        sub_8CC0D0(v17, v16);
        return sub_899FE0(v16, v17);
      }
      return sub_8CA1D0(v6, a1);
    }
    v18 = *(_QWORD *)(a1 + 88);
    v19 = sub_8C6880(v6, a1, v8, v9);
    if ( !v19 )
      return sub_8CA1D0(v6, a1);
    result = v19[1];
    if ( result != a1 )
      return sub_8CA500(v18, *(_QWORD *)(result + 88));
  }
  else if ( unk_4D03FC0 )
  {
    if ( (unsigned __int8)(result - 4) <= 1u )
    {
      return sub_8CCC20(a1);
    }
    else if ( (unsigned __int8)(result - 10) <= 1u || (_BYTE)result == 17 )
    {
      return (__int64)sub_8CC1D0(a1);
    }
    else if ( (_BYTE)result == 3 )
    {
      return (__int64)sub_8C9210(a1);
    }
    else if ( (_BYTE)result == 7 )
    {
      return (__int64)sub_8CC330(a1);
    }
  }
  return result;
}
