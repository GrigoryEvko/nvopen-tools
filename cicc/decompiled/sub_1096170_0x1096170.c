// Function: sub_1096170
// Address: 0x1096170
//
__int64 __fastcall sub_1096170(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // rax
  char v4; // r8
  char v5; // r9
  int v6; // ecx
  __int64 v7; // rdi
  _BYTE *v8; // rdx
  unsigned __int8 *v10; // rdx
  __int64 v11; // rdi

  v3 = *(unsigned __int8 **)(a2 + 152);
  v4 = *(_BYTE *)(a2 + 114);
  v5 = *(_BYTE *)(a2 + 113);
  v6 = *v3;
  if ( *(v3 - 1) == 46 && (unsigned __int8)(v6 - 48) <= 9u )
  {
    v10 = v3 + 1;
    do
    {
      *(_QWORD *)(a2 + 152) = v10;
      v6 = *v10;
      v3 = v10++;
    }
    while ( (unsigned __int8)(v6 - 48) <= 9u );
    if ( (unsigned __int8)((v6 & 0xDF) - 65) <= 0x19u )
      goto LABEL_27;
    if ( (unsigned __int8)(v6 - 36) > 0x3Bu )
      goto LABEL_22;
    v11 = 0x800000008000401LL;
    if ( _bittest64(&v11, (unsigned int)(v6 - 36)) )
    {
LABEL_27:
      if ( (v6 & 0xDF) == 0x45 )
      {
LABEL_28:
        sub_1095CD0(a1, a2);
        return a1;
      }
      goto LABEL_2;
    }
    if ( (_BYTE)v6 != 64 || !v5 )
    {
LABEL_22:
      if ( (_BYTE)v6 != 35 || !v4 )
        goto LABEL_28;
    }
  }
LABEL_2:
  v7 = 0x8000000083FF401LL;
  while ( 1 )
  {
    if ( (unsigned __int8)((v6 & 0xDF) - 65) <= 0x19u )
      goto LABEL_8;
    if ( (unsigned __int8)(v6 - 36) > 0x3Bu )
      break;
    if ( !_bittest64(&v7, (unsigned int)(v6 - 36)) && ((_BYTE)v6 != 64 || !v5) )
      goto LABEL_11;
LABEL_8:
    *(_QWORD *)(a2 + 152) = ++v3;
    v6 = *v3;
  }
  if ( (_BYTE)v6 == 35 && v4 )
    goto LABEL_8;
LABEL_11:
  v8 = *(_BYTE **)(a2 + 104);
  if ( v3 == v8 + 1 && *v8 == 46 )
  {
    *(_DWORD *)a1 = 25;
    *(_QWORD *)(a1 + 8) = v8;
    *(_QWORD *)(a1 + 16) = 1;
    *(_DWORD *)(a1 + 32) = 64;
    *(_QWORD *)(a1 + 24) = 0;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v8;
    *(_DWORD *)a1 = 2;
    *(_QWORD *)(a1 + 16) = v3 - v8;
    *(_DWORD *)(a1 + 32) = 64;
    *(_QWORD *)(a1 + 24) = 0;
  }
  return a1;
}
