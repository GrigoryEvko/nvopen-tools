// Function: sub_254F6C0
// Address: 0x254f6c0
//
__int64 __fastcall sub_254F6C0(__int64 a1, __int64 *a2, char *a3)
{
  __int64 result; // rax
  unsigned __int8 *v5; // r14
  unsigned __int8 v6; // cl
  unsigned __int64 v7; // rdx
  unsigned __int8 *v8; // r15
  unsigned __int8 v9; // cl
  char v10; // al

  result = 0;
  if ( *(_DWORD *)(a1 + 3556) <= dword_4FEEF68[0] )
  {
    if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) <= 1 )
      goto LABEL_15;
    v5 = sub_250CBE0(a2, (__int64)a2);
    v6 = sub_2509800(a2);
    if ( v6 <= 7u && ((1LL << v6) & 0xA8) != 0 )
    {
      v7 = *a2 & 0xFFFFFFFFFFFFFFFCLL;
      if ( (*a2 & 3) == 3 )
        v7 = *(_QWORD *)(v7 + 24);
      if ( **(_BYTE **)(v7 - 32) == 25 )
        goto LABEL_15;
    }
    v8 = sub_250CBE0(a2, (__int64)a2);
    v9 = sub_2509800(a2);
    if ( v9 <= 6u && ((1LL << v9) & 0x54) != 0 && !(unsigned __int8)sub_254F400(a1, v8) )
    {
LABEL_15:
      v10 = 0;
    }
    else
    {
      v10 = 1;
      if ( v5 )
      {
        v10 = *(_BYTE *)(a1 + 4296);
        if ( !v10 )
        {
          v10 = sub_253A110(*(_QWORD *)(a1 + 200), (__int64)v5);
          if ( !v10 )
            v10 = sub_254D3B0(a1, a2);
        }
      }
    }
    *a3 = v10;
    return 1;
  }
  return result;
}
