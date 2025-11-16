// Function: sub_730E80
// Address: 0x730e80
//
unsigned __int64 __fastcall sub_730E80(__int64 a1)
{
  _QWORD *i; // rbx
  _QWORD *k; // rax
  unsigned __int64 v4; // r8
  __int64 v6; // rdi
  char j; // al
  __int64 v8; // r8
  __int64 **v9; // rax
  unsigned __int64 v10; // rdx

  for ( i = *(_QWORD **)(a1 + 160); i; i = (_QWORD *)i[14] )
  {
    if ( !i[14] )
    {
      v6 = i[15];
      for ( j = *(_BYTE *)(v6 + 140); j == 12; j = *(_BYTE *)(v6 + 140) )
        v6 = *(_QWORD *)(v6 + 160);
      if ( (unsigned __int8)(j - 9) <= 2u )
        v8 = sub_730E80(v6);
      else
        v8 = *(_QWORD *)(v6 + 128);
      v4 = i[16] + v8;
      if ( (*(_BYTE *)(a1 + 176) & 0x10) != 0 )
      {
        v9 = *(__int64 ***)(a1 + 168);
        while ( 1 )
        {
          v9 = (__int64 **)*v9;
          if ( !v9 )
            break;
          if ( ((_BYTE)v9[12] & 0x22) == 2 )
          {
            v10 = (unsigned __int64)v9[13];
            if ( v10 >= v4 )
              v4 = *(_QWORD *)(v9[5][21] + 32) + v10;
          }
        }
      }
      return v4;
    }
  }
  for ( k = **(_QWORD ***)(a1 + 168); k; k = (_QWORD *)*k )
  {
    if ( !*k )
      return *(_QWORD *)(k[5] + 128LL) + k[13];
  }
  v4 = 0;
  if ( (*(_BYTE *)(a1 + 176) & 0x50) != 0 )
    return unk_4F06A68;
  return v4;
}
