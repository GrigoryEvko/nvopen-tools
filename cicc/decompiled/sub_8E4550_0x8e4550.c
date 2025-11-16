// Function: sub_8E4550
// Address: 0x8e4550
//
__int64 __fastcall sub_8E4550(__int64 *a1, __int64 a2)
{
  _BYTE *v2; // r12
  __int64 result; // rax
  unsigned __int8 v4; // cl
  char v6; // cl
  __int64 i; // rdi
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rdi
  char j; // al
  __int64 **v12; // rbx
  __int64 *v13; // rdi

  v2 = *(_BYTE **)(*a1 + 96);
  if ( (v2[179] & 2) != 0 )
    return v2[179] & 1;
  if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774 )
  {
    v4 = v2[178] >> 7;
    v2[179] = v4 | v2[179] & 0xFC | 2;
    return v4;
  }
  if ( (char)v2[181] < 0 && (unsigned int)sub_8E43E0(a1, a2) )
  {
    for ( i = a1[20]; ; i = *(_QWORD *)(v9 + 112) )
    {
      v8 = sub_72FD90(i, 7);
      v9 = v8;
      if ( !v8 )
        break;
      v10 = sub_8D4130(*(_QWORD *)(v8 + 120));
      for ( j = *(_BYTE *)(v10 + 140); j == 12; j = *(_BYTE *)(v10 + 140) )
        v10 = *(_QWORD *)(v10 + 160);
      if ( (unsigned __int8)(j - 9) <= 2u && !(unsigned int)sub_8E4550(v10) )
        goto LABEL_8;
    }
    v12 = *(__int64 ***)a1[21];
    if ( !v12 )
    {
LABEL_25:
      v6 = 1;
      result = 1;
      goto LABEL_9;
    }
    while ( 1 )
    {
      if ( ((_BYTE)v12[12] & 1) != 0 )
      {
        v13 = v12[5];
        if ( (unsigned __int8)(*((_BYTE *)v13 + 140) - 9) <= 2u && !(unsigned int)sub_8E4550(v13) )
          break;
      }
      v12 = (__int64 **)*v12;
      if ( !v12 )
        goto LABEL_25;
    }
  }
LABEL_8:
  v6 = 0;
  result = 0;
LABEL_9:
  v2[179] = v6 | 2 | v2[179] & 0xFC;
  return result;
}
