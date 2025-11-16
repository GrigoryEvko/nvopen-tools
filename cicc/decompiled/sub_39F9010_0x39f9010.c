// Function: sub_39F9010
// Address: 0x39f9010
//
__int64 __fastcall sub_39F9010(__int64 a1, __int64 a2, __int64 a3)
{
  __int16 v4; // ax
  unsigned __int8 v5; // dl
  char *v6; // r12
  __int64 result; // rax
  unsigned __int64 v8; // [rsp+0h] [rbp-28h] BYREF
  unsigned __int64 v9[4]; // [rsp+8h] [rbp-20h] BYREF

  v4 = *(_WORD *)(a1 + 32) >> 3;
  if ( (_BYTE)v4 == 0xFF )
    goto LABEL_10;
  v5 = v4 & 0x70;
  if ( (v4 & 0x70) != 0x20 )
  {
    if ( v5 > 0x20u )
    {
      if ( v5 == 48 )
      {
        v6 = *(char **)(a1 + 16);
        goto LABEL_6;
      }
      if ( v5 == 80 )
        goto LABEL_10;
LABEL_14:
      abort();
    }
    if ( (v4 & 0x60) != 0 )
      goto LABEL_14;
LABEL_10:
    v6 = 0;
    goto LABEL_6;
  }
  v6 = *(char **)(a1 + 8);
LABEL_6:
  sub_39F8BA0(v4, v6, (char *)(a2 + 8), &v8);
  sub_39F8BA0(*(_WORD *)(a1 + 32) >> 3, v6, (char *)(a3 + 8), v9);
  result = 1;
  if ( v8 <= v9[0] )
    return (unsigned int)-(v8 < v9[0]);
  return result;
}
