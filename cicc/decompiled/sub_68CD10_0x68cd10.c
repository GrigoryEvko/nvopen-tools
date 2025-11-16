// Function: sub_68CD10
// Address: 0x68cd10
//
__int64 __fastcall sub_68CD10(__int64 *a1, int a2)
{
  __int64 result; // rax
  char v4; // cl
  __int64 v5; // rdi
  char v6; // dl
  __int64 v7; // r12
  char v8; // al
  _DWORD v9[9]; // [rsp+Ch] [rbp-24h] BYREF

  v9[0] = a2;
  result = sub_6F6BD0(a1, 0);
  if ( a2 )
    return result;
  result = unk_4D03C50;
  if ( (*(_BYTE *)(unk_4D03C50 + 17LL) & 0x20) != 0 )
    return result;
  v4 = *((_BYTE *)a1 + 16);
  if ( v4 == 1 )
  {
    v7 = a1[18];
    if ( *(_BYTE *)(v7 + 24) != 1 )
      goto LABEL_35;
    do
    {
      v8 = *(_BYTE *)(v7 + 56);
      if ( v8 == 5 )
      {
        result = sub_8D2600(*(_QWORD *)v7);
        if ( (_DWORD)result )
          return result;
        v8 = *(_BYTE *)(v7 + 56);
      }
      if ( v8 != 91 )
        break;
      if ( (unsigned int)sub_731770(*(_QWORD *)(v7 + 72), v9) )
        goto LABEL_25;
      v7 = *(_QWORD *)(*(_QWORD *)(v7 + 72) + 16LL);
    }
    while ( *(_BYTE *)(v7 + 24) == 1 );
    if ( !v9[0] )
    {
LABEL_35:
      if ( (unsigned int)sub_731770(v7, v9) )
LABEL_25:
        v9[0] = 1;
    }
    if ( (*(_BYTE *)(unk_4D03C50 + 20LL) & 0x40) != 0 )
      sub_68CA80(v7);
    result = v9[0];
    if ( !v9[0] )
    {
      v5 = *a1;
LABEL_11:
      result = sub_8D2600(v5);
      if ( !(_DWORD)result && !v9[0] )
      {
        result = sub_6E53E0(5, 174, (char *)a1 + 68);
        if ( (_DWORD)result )
          return sub_684B30(0xAEu, (_DWORD *)a1 + 17);
      }
    }
  }
  else if ( v4 )
  {
    v5 = *a1;
    v6 = *(_BYTE *)(*a1 + 140);
    if ( v6 == 12 )
    {
      result = *a1;
      do
      {
        result = *(_QWORD *)(result + 160);
        v6 = *(_BYTE *)(result + 140);
      }
      while ( v6 == 12 );
    }
    if ( v6 && (v4 != 2 || *((_BYTE *)a1 + 317) != 12 && (*((_BYTE *)a1 + 315) & 2) == 0) )
      goto LABEL_11;
  }
  return result;
}
