// Function: sub_2E2EB00
// Address: 0x2e2eb00
//
__int64 __fastcall sub_2E2EB00(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // edx
  char v5; // cl
  int v6; // eax
  char v7; // dl
  __int64 result; // rax
  char *v9; // rax
  size_t v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  bool v14; // zf
  char v15; // cl

  v4 = *(_BYTE *)(a2 + 32) & 0xF;
  v5 = *(_BYTE *)(a2 + 32) & 0xF;
  if ( (unsigned int)(v4 - 7) > 1 )
  {
    *(_BYTE *)(a3 + 32) = v5 | *(_BYTE *)(a3 + 32) & 0xF0;
  }
  else
  {
    *(_WORD *)(a3 + 32) = *(_BYTE *)(a2 + 32) & 0xF | *(_WORD *)(a3 + 32) & 0xFCC0;
    if ( v4 == 7 )
      goto LABEL_3;
  }
  if ( v4 != 8 )
  {
    v14 = v5 == 9;
    v15 = *(_BYTE *)(a3 + 32);
    if ( (v15 & 0x30) != 0 && !v14 )
    {
      v6 = *(unsigned __int8 *)(a3 + 33) | 0x40;
      *(_BYTE *)(a3 + 33) |= 0x40u;
      *(_BYTE *)(a3 + 32) = *(_BYTE *)(a2 + 32) & 0x30 | v15 & 0xCF;
      if ( v4 == 7 )
        goto LABEL_4;
    }
    else
    {
      *(_BYTE *)(a3 + 32) = *(_BYTE *)(a2 + 32) & 0x30 | *(_BYTE *)(a3 + 32) & 0xCF;
      v6 = *(unsigned __int8 *)(a3 + 33);
    }
    if ( (*(_BYTE *)(a3 + 32) & 0x30) == 0 || v14 )
      goto LABEL_5;
    goto LABEL_4;
  }
LABEL_3:
  v6 = *(unsigned __int8 *)(a3 + 33) | 0x40;
  v7 = *(_BYTE *)(a3 + 32) & 0xCF;
  *(_BYTE *)(a3 + 33) |= 0x40u;
  *(_BYTE *)(a3 + 32) = *(_BYTE *)(a2 + 32) & 0x30 | v7;
LABEL_4:
  v6 |= 0x40u;
  *(_BYTE *)(a3 + 33) = v6;
LABEL_5:
  result = *(_BYTE *)(a2 + 33) & 0x40 | v6 & 0xFFFFFFBF;
  *(_BYTE *)(a3 + 33) = result;
  if ( *(_QWORD *)(a2 + 48) )
  {
    v9 = (char *)sub_BD5D20(a3);
    v11 = sub_BAA410(a1, v9, v10);
    sub_B2F990(a3, v11, v12, v13);
    result = *(_QWORD *)(a3 + 48);
    *(_DWORD *)(result + 8) = *(_DWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  }
  return result;
}
