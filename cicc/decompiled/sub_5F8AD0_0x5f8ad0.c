// Function: sub_5F8AD0
// Address: 0x5f8ad0
//
__int64 __fastcall sub_5F8AD0(__int64 a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  __int64 v5; // rdx
  __int64 result; // rax
  bool v7; // r13
  __int64 v9; // r12
  __int64 v10; // r8
  char v11; // al
  __int64 j; // rdi
  int v13; // r12d
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // [rsp+0h] [rbp-70h]
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 i; // [rsp+18h] [rbp-58h]
  bool v24; // [rsp+2Eh] [rbp-42h]
  char v25; // [rsp+2Fh] [rbp-41h]
  _DWORD v26[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v5 = 0;
  v21 = *(_QWORD *)(a1 + 152);
  v25 = *(_BYTE *)(a1 + 174);
  if ( (*(_BYTE *)(a1 + 194) & 0x40) == 0 )
    v5 = **(_QWORD **)(v21 + 168);
  result = a1 + 64;
  v24 = **(_QWORD **)(v21 + 168) == 0;
  v20 = v5;
  v7 = *(_BYTE *)(a1 + 174) == 2;
  for ( i = a1 + 64; a3; a3 = (__int64 *)a3[14] )
  {
    v9 = *a3;
    if ( (a3[18] & 0x10) != 0 && v7 )
    {
      v19 = *(_QWORD *)(a3[15] + 160);
      if ( v19 )
        sub_5F8AD0(a1, a2, v19, a4);
    }
    if ( v9 && (*(_BYTE *)(v9 + 81) & 0x20) != 0 )
    {
LABEL_30:
      *a4 = 1;
      return (__int64)a4;
    }
    v10 = a3[15];
    v11 = *(_BYTE *)(v10 + 140);
    for ( j = v10; v11 == 12; v11 = *(_BYTE *)(j + 140) )
      j = *(_QWORD *)(j + 160);
    if ( v11 == 8 )
      v10 = sub_8D40F0(j);
    v13 = 0;
    v14 = sub_8D21C0(v10);
    if ( (*(_BYTE *)(v14 + 140) & 0xFB) == 8 )
      v13 = sub_8D4C10(v14, dword_4F077C4 != 2);
    if ( (*((_BYTE *)a3 + 145) & 0x20) != 0 && (*(_BYTE *)(a1 + 194) & 0x40) == 0 && v25 == 1 && v24 )
    {
      if ( !a3[19] )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a2 + 96LL) + 183LL) & 0x10) != 0 || (sub_5E87D0(a2), !a3[19]) )
        {
          sub_6851C0(2438, i);
          sub_7296C0(v26);
          a3[19] = sub_72C9D0();
          sub_729730(v26[0]);
          *a4 = 1;
          return (__int64)a4;
        }
      }
      result = sub_731C20();
      if ( (_DWORD)result )
        goto LABEL_30;
    }
    else
    {
      if ( (unsigned int)sub_8DBE70(v14) )
        goto LABEL_30;
      result = (unsigned int)*(unsigned __int8 *)(v14 + 140) - 9;
      if ( (unsigned __int8)(*(_BYTE *)(v14 + 140) - 9) <= 2u && *(_QWORD *)v14 )
      {
        v26[0] = 0;
        result = sub_8D23B0(v14);
        if ( (_DWORD)result
          || (result = (unsigned int)*(unsigned __int8 *)(v14 + 140) - 9,
              (unsigned __int8)(*(_BYTE *)(v14 + 140) - 9) > 2u) )
        {
          if ( v26[0] )
            goto LABEL_30;
        }
        else
        {
          v17 = sub_5E8390(v14, v25, v20, v13, i, (int)v26);
          result = v26[0];
          if ( v26[0] )
            goto LABEL_30;
          if ( v17 )
          {
            result = sub_5F8900(v17, v21, v15, v16, v18);
            if ( (_DWORD)result )
              goto LABEL_30;
          }
        }
      }
    }
  }
  return result;
}
