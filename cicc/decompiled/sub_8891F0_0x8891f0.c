// Function: sub_8891F0
// Address: 0x8891f0
//
__int64 __fastcall sub_8891F0(unsigned __int16 a1, char *a2, int a3, unsigned __int16 a4)
{
  int v8; // r8d
  __int64 result; // rax
  char *v11; // rcx
  char *v12; // rdx
  __int64 v13; // rsi
  char *v14; // rax
  _DWORD *v15; // r14
  unsigned int v16; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v17; // [rsp+4h] [rbp-2Ch] BYREF
  char *v18[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( dword_4D0455C && unk_4D04600 > 0x2E693u && unk_4D045F8 )
  {
    v11 = (char *)&unk_4B7EE2C;
    v12 = (char *)&unk_4B7EE2C - 3468;
    while ( 1 )
    {
      if ( v12 >= v11 )
      {
        v8 = 0;
        if ( a2 )
          goto LABEL_4;
        goto LABEL_26;
      }
      v13 = (v11 - v12) >> 1;
      v14 = &v12[(v13 + ((unsigned __int64)(v11 - v12) >> 63)) & 0xFFFFFFFFFFFFFFFELL];
      if ( *(_WORD *)v14 == a4 )
        break;
      if ( *(_WORD *)v14 <= a4 )
        v12 = v14 + 2;
      else
        v11 = &v12[(v13 + ((unsigned __int64)(v11 - v12) >> 63)) & 0xFFFFFFFFFFFFFFFELL];
    }
    if ( a2 )
    {
      v8 = 1;
      goto LABEL_4;
    }
    v16 = 0;
    v17 = 0;
    sub_888610(off_4AE3B40[a1], &v16, (int *)&v17, v18, 1);
    if ( !a3 )
      goto LABEL_5;
LABEL_18:
    result = v17;
    goto LABEL_6;
  }
  v8 = 0;
  if ( a2 )
  {
LABEL_4:
    v16 = 0;
    v17 = 0;
    sub_888610(a2, &v16, (int *)&v17, v18, v8);
    if ( !a3 )
    {
LABEL_5:
      result = v16;
      goto LABEL_6;
    }
    goto LABEL_18;
  }
LABEL_26:
  v15 = (_DWORD *)(unk_4D03FB0 + 24LL * a1);
  if ( !v15[2] )
  {
    sub_888610(off_4AE3B40[a1], v15 + 3, v15 + 4, (char **)(unk_4D03FB0 + 24LL * a1), 0);
    v15[2] = 1;
  }
  if ( a3 )
    result = (unsigned int)v15[4];
  else
    result = (unsigned int)v15[3];
LABEL_6:
  if ( (_DWORD)result )
  {
    if ( a4 == 24961 || a4 == 24956 )
    {
      if ( dword_4D0455C )
        return 0;
    }
    else if ( (unsigned __int16)(a4 - 24990) <= 1u )
    {
      if ( !dword_4D0455C )
        return 0;
    }
    else if ( dword_4D0455C && unk_4D04600 > 0x2E693u && unk_4D045F8 )
    {
      if ( a4 == 6146 || a4 == 8114 )
        return 0;
    }
    else if ( (unsigned __int16)(a4 - 25773) <= 1u )
    {
      return 0;
    }
  }
  return result;
}
