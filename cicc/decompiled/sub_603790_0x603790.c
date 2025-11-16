// Function: sub_603790
// Address: 0x603790
//
__int64 __fastcall sub_603790(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  _QWORD *v3; // r12
  __int64 v4; // rbx
  int v5; // r13d
  __int64 v6; // rdi
  __int64 v7; // rbx
  int v8; // r13d
  __int64 v9; // rdx
  __int64 i; // rbx
  _DWORD v11[9]; // [rsp+Ch] [rbp-24h] BYREF

  result = (__int64)&unk_4F07280;
  v3 = (_QWORD *)unk_4F07288;
  v4 = *(_QWORD *)(unk_4F07288 + 104LL);
  if ( !v4 )
    return result;
  v5 = 0;
  do
  {
    if ( (*(_BYTE *)(v4 + 89) & 1) != 0 )
      goto LABEL_6;
    result = *(unsigned __int8 *)(v4 + 140);
    if ( (unsigned __int8)(result - 9) <= 2u )
    {
      result = sub_5E7150(v4);
      if ( !(_DWORD)result )
        goto LABEL_6;
      v6 = *(_QWORD *)(v4 + 168);
      if ( *(_QWORD *)(v6 + 168)
        || (*(_BYTE *)(*(_QWORD *)v4 + 82LL) & 2) != 0
        || (result = sub_5E8C70(v6, (__int64)a2), (_DWORD)result) )
      {
        v11[0] = 0;
        a2 = v11;
        *(_BYTE *)(v4 + 88) = *(_BYTE *)(v4 + 88) & 0x8F | 0x20;
        result = sub_5E7470((_QWORD *)v4, v11);
        goto LABEL_6;
      }
LABEL_32:
      v5 = 1;
      goto LABEL_6;
    }
    if ( (_BYTE)result == 2 && (*(_BYTE *)(v4 + 161) & 8) != 0 )
    {
      result = sub_5E7150(v4);
      if ( (_DWORD)result )
      {
        result = *(_QWORD *)v4;
        if ( *(_QWORD *)v4 && (*(_BYTE *)(result + 82) & 2) != 0 )
        {
          result = *(_BYTE *)(v4 + 88) & 0x8F | 0x20u;
          *(_BYTE *)(v4 + 88) = *(_BYTE *)(v4 + 88) & 0x8F | 0x20;
          goto LABEL_6;
        }
        goto LABEL_32;
      }
    }
LABEL_6:
    v4 = *(_QWORD *)(v4 + 112);
  }
  while ( v4 );
  if ( v5 )
  {
    v9 = v3[13];
    if ( v9 )
    {
      v8 = 0;
      do
      {
        if ( (*(_BYTE *)(v9 + 89) & 1) == 0 )
        {
          result = *(unsigned __int8 *)(v9 + 140);
          if ( (unsigned __int8)(result - 9) <= 2u || (_BYTE)result == 2 && (*(_BYTE *)(v9 + 161) & 8) != 0 )
          {
            result = sub_5E7150(v9);
            v8 -= ((_DWORD)result == 0) - 1;
          }
        }
        v9 = *(_QWORD *)(v9 + 112);
      }
      while ( v9 );
      if ( v8 )
      {
        v7 = v3[14];
        if ( v7 )
        {
          while ( 1 )
          {
            if ( (*(_BYTE *)(v7 + 88) & 0x60) != 0 )
            {
              v11[0] = 0;
              result = sub_5E71C0(*(_QWORD *)(v7 + 120), v11);
              v8 -= v11[0];
              if ( v8 <= 0 )
                break;
            }
            v7 = *(_QWORD *)(v7 + 112);
            if ( !v7 )
              goto LABEL_35;
          }
        }
        else
        {
LABEL_35:
          for ( i = v3[18]; i; i = *(_QWORD *)(i + 112) )
          {
            if ( (*(_BYTE *)(i + 88) & 0x60) != 0 )
            {
              v11[0] = 0;
              result = sub_5E71C0(*(_QWORD *)(i + 152), v11);
              v8 -= v11[0];
              if ( v8 <= 0 )
                break;
            }
          }
        }
      }
    }
  }
  return result;
}
