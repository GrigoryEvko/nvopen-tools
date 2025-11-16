// Function: sub_92A460
// Address: 0x92a460
//
__int64 __fastcall sub_92A460(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  __int64 v7; // rcx
  int v8; // eax
  unsigned int **v9; // rdi
  _BYTE *v10; // r12
  int v11; // eax
  unsigned int **v13; // rdi
  char v14; // r9
  unsigned int v15; // [rsp+8h] [rbp-58h]
  _QWORD v16[4]; // [rsp+10h] [rbp-50h] BYREF
  char v17; // [rsp+30h] [rbp-30h]
  char v18; // [rsp+31h] [rbp-2Fh]

  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v8 - 17) <= 1 )
    LOBYTE(v8) = *(_BYTE *)(**(_QWORD **)(v7 + 16) + 8LL);
  if ( (unsigned __int8)v8 <= 3u || (_BYTE)v8 == 5 || (v8 & 0xFD) == 4 )
  {
    v9 = *(unsigned int ***)(a1 + 8);
    v18 = 1;
    v17 = 3;
    v16[0] = "add";
    v10 = (_BYTE *)sub_92A220(v9, (_BYTE *)a2, a3, v15, (__int64)v16, 0);
    if ( unk_4D04700 )
    {
      if ( *v10 > 0x1Cu )
      {
        v11 = sub_B45210(v10);
        sub_B45150(v10, v11 | 1u);
      }
    }
    return (__int64)v10;
  }
  else
  {
    v18 = 1;
    if ( (unsigned __int8)sub_91B6F0(a4) )
    {
      v13 = *(unsigned int ***)(a1 + 8);
      v16[0] = "add";
      v14 = 1;
      v17 = 3;
    }
    else
    {
      v13 = *(unsigned int ***)(a1 + 8);
      v14 = 0;
      v17 = 3;
      v16[0] = "add";
    }
    return sub_929C50(v13, (_BYTE *)a2, a3, (__int64)v16, 0, v14);
  }
}
