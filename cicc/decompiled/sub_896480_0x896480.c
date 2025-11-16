// Function: sub_896480
// Address: 0x896480
//
__int64 __fastcall sub_896480(__int64 a1, __int64 a2, int a3)
{
  char v4; // al
  __int64 v5; // r15
  __int64 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 v10; // rbx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // r13
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 *v16; // r15
  __int64 v17; // rdx

  v4 = *(_BYTE *)(a1 + 80);
  v5 = *(_QWORD *)(a2 + 96);
  if ( v4 == 9 || v4 == 7 )
  {
    v6 = *(__int64 **)(a1 + 88);
  }
  else
  {
    v6 = 0;
    if ( v4 == 21 )
      v6 = *(__int64 **)(*(_QWORD *)(a1 + 88) + 192LL);
  }
  v7 = v6[15];
  v8 = *(_QWORD *)(a2 + 88);
  if ( *(_BYTE *)(v7 + 140) == 11 && sub_879510(*(_QWORD **)v7) )
  {
    result = *(_QWORD *)(*(_QWORD *)(v8 + 168) + 152LL);
    v16 = *(__int64 **)(result + 112);
    if ( !v16 )
      return result;
    while ( 1 )
    {
      v10 = *v16;
      result = sub_892240(*v16);
      if ( result )
      {
        switch ( *(_BYTE *)(v10 + 80) )
        {
          case 4:
          case 5:
            result = *(_QWORD *)(*(_QWORD *)(v10 + 96) + 80LL);
            break;
          case 6:
            result = *(_QWORD *)(*(_QWORD *)(v10 + 96) + 32LL);
            break;
          case 9:
          case 0xA:
            result = *(_QWORD *)(*(_QWORD *)(v10 + 96) + 56LL);
            break;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            result = *(_QWORD *)(v10 + 88);
            break;
          default:
            BUG();
        }
        if ( *(_DWORD *)(result + 64) == a3 )
          break;
      }
      v16 = (__int64 *)v16[14];
      if ( !v16 )
        return result;
    }
  }
  else
  {
    result = *(_QWORD *)(*(_QWORD *)(v8 + 168) + 152LL);
    if ( !result )
      return result;
    if ( (*(_BYTE *)(result + 29) & 0x20) != 0 )
      return result;
    result = sub_883800(v5 + 192, *(_QWORD *)a1);
    v10 = result;
    if ( !result )
      return result;
    while ( 1 )
    {
      result = *(unsigned __int8 *)(v10 + 80);
      if ( (_BYTE)result == *(_BYTE *)(a1 + 80) )
      {
        if ( (_BYTE)result == 21 )
          goto LABEL_43;
        result = sub_892240(v10);
        if ( result )
          break;
      }
      v10 = *(_QWORD *)(v10 + 32);
      if ( !v10 )
        return result;
    }
    result = *(unsigned __int8 *)(v10 + 80);
    if ( (unsigned __int8)result <= 0xAu )
      goto LABEL_16;
    if ( (unsigned __int8)(result - 19) > 3u )
    {
      v17 = 0;
      goto LABEL_35;
    }
LABEL_43:
    v17 = *(_QWORD *)(v10 + 88);
LABEL_35:
    if ( (_BYTE)result == 21 && *(_DWORD *)(v17 + 64) != a3 )
    {
      v10 = *(_QWORD *)(v17 + 144);
      if ( !v10 )
        return result;
      while ( 1 )
      {
        result = *(_QWORD *)(v10 + 88);
        if ( *(_DWORD *)(result + 64) == a3 )
          break;
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
          return result;
      }
    }
  }
LABEL_16:
  v11 = (_QWORD *)sub_880C60();
  if ( *(_BYTE *)(a1 + 80) == 9 )
  {
    *(_QWORD *)(a1 + 96) = v11;
    v11[3] = a1;
  }
  else
  {
    v12 = *v6;
    *(_QWORD *)(v12 + 96) = v11;
    v11[3] = v12;
  }
  v11[4] = v10;
  if ( *(_BYTE *)(a1 + 80) == 9 )
  {
    v13 = *(_QWORD **)(*(_QWORD *)(v10 + 96) + 56LL);
    *v11 = v13[22];
    v13[22] = v11;
    *((_BYTE *)v6 + 170) |= 0x10u;
    result = (__int64)sub_725CE0();
    v6[27] = result;
  }
  else
  {
    v13 = *(_QWORD **)(a1 + 88);
    *(_BYTE *)(a1 + 81) = *(_BYTE *)(v10 + 81) & 2 | *(_BYTE *)(a1 + 81) & 0xFD;
    v14 = *(_QWORD *)(v10 + 88);
    v13[11] = v10;
    v15 = sub_878440();
    v15[1] = a1;
    *v15 = *(_QWORD *)(v14 + 96);
    *(_QWORD *)(v14 + 96) = v15;
    result = v6[27];
  }
  *(_QWORD *)(result + 16) = v13[13];
  return result;
}
