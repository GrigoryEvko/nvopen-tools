// Function: sub_25DD4B0
// Address: 0x25dd4b0
//
__int64 __fastcall sub_25DD4B0(__int64 a1, __int64 a2, void (__fastcall *a3)(__int64, __int64), __int64 a4)
{
  __int64 v5; // r13
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned int v11; // r8d
  _QWORD *v13; // rdx
  _QWORD *v14; // rcx

  v5 = a2;
  sub_AD0030(a1);
  v7 = *(_BYTE *)(a1 + 32) & 0xF;
  if ( ((v7 + 15) & 0xFu) > 2 && ((v7 + 9) & 0xFu) > 1 && !sub_B2FC80(a1) )
    return 0;
  v8 = sub_B326A0(a1);
  if ( v8 && (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
  {
    if ( *(_BYTE *)(a2 + 28) )
    {
      v13 = *(_QWORD **)(a2 + 8);
      v14 = &v13[*(unsigned int *)(a2 + 20)];
      if ( v13 != v14 )
      {
        while ( v8 != *v13 )
        {
          if ( v14 == ++v13 )
            goto LABEL_5;
        }
        return 0;
      }
    }
    else
    {
      a2 = v8;
      if ( sub_C8CA60(v5, v8) )
        return 0;
    }
  }
LABEL_5:
  if ( !*(_BYTE *)a1 )
  {
    if ( (!sub_B2FC80(a1) || *(_QWORD *)(a1 + 16)) && !(unsigned __int8)sub_B2E360(a1, a2, v9, v10, v11) )
      return 0;
    if ( !*(_BYTE *)a1 )
    {
      if ( a3 )
        a3(a4, a1);
    }
LABEL_12:
    sub_BA58F0(a1);
    sub_B30810((_QWORD *)a1);
    return 1;
  }
  if ( !*(_QWORD *)(a1 + 16) )
    goto LABEL_12;
  return 0;
}
