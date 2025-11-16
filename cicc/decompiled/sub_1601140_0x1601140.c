// Function: sub_1601140
// Address: 0x1601140
//
__int64 __fastcall sub_1601140(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // r13
  int v4; // eax
  int v5; // edx
  unsigned int v6; // r13d
  __int64 v7; // rdx
  __int64 v8; // r13

  if ( *(char *)(a1 + 23) < 0 )
  {
    v1 = sub_1648A40(a1);
    v3 = v1 + v2;
    if ( *(char *)(a1 + 23) >= 0 )
    {
      v7 = 0;
      if ( (unsigned int)(v3 >> 4) )
        goto LABEL_7;
    }
    else if ( (unsigned int)((v3 - sub_1648A40(a1)) >> 4) )
    {
      if ( *(char *)(a1 + 23) >= 0 )
      {
        v7 = 0;
      }
      else
      {
        v4 = sub_1648A40(a1);
        v6 = v4 + v5;
        if ( *(char *)(a1 + 23) >= 0 )
          v7 = v6;
        else
          v7 = v6 - (unsigned int)sub_1648A40(a1);
      }
LABEL_7:
      v8 = sub_1648AB0(72, *(_DWORD *)(a1 + 20) & 0xFFFFFFF, v7);
      if ( !v8 )
        return v8;
      goto LABEL_11;
    }
  }
  v8 = sub_1648A60(72, *(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( v8 )
LABEL_11:
    sub_15F5F30(v8, a1);
  return v8;
}
