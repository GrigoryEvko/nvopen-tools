// Function: sub_1C2ECF0
// Address: 0x1c2ecf0
//
__int64 __fastcall sub_1C2ECF0(__int64 a1, __int64 a2)
{
  char v2; // dl
  int v3; // eax
  int v5; // [rsp+8h] [rbp-28h] BYREF
  char v6; // [rsp+Ch] [rbp-24h]
  int v7; // [rsp+10h] [rbp-20h] BYREF
  char v8; // [rsp+14h] [rbp-1Ch]
  int v9; // [rsp+18h] [rbp-18h] BYREF
  char v10; // [rsp+1Ch] [rbp-14h]

  sub_1C2EC00((__int64)&v5, a2);
  sub_1C2EC50((__int64)&v7, a2);
  sub_1C2ECA0((__int64)&v9, a2);
  if ( v6 )
  {
    v3 = v5;
    v2 = v10;
    if ( !v8 )
    {
      if ( !v10 )
        goto LABEL_5;
      goto LABEL_11;
    }
LABEL_4:
    v3 *= v7;
    if ( !v2 )
    {
LABEL_5:
      *(_DWORD *)a1 = v3;
      *(_BYTE *)(a1 + 4) = 1;
      return a1;
    }
LABEL_11:
    v3 *= v9;
    goto LABEL_5;
  }
  if ( v8 )
  {
    v2 = v10;
    v3 = 1;
    goto LABEL_4;
  }
  if ( v10 )
  {
    v3 = 1;
    goto LABEL_11;
  }
  *(_BYTE *)(a1 + 4) = 0;
  return a1;
}
