// Function: sub_315F020
// Address: 0x315f020
//
int __fastcall sub_315F020(__int64 a1, __int64 a2)
{
  const char *v2; // r13
  const char *v3; // r12
  char v5; // al
  __int64 v6; // rdi
  __int64 v7; // rdi
  const char *v9[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v10; // [rsp+20h] [rbp-30h]

  v3 = (const char *)a2;
  sub_315E1E0(a1, (const char **)a2);
  v5 = *(_BYTE *)(a2 + 32);
  if ( v5 )
  {
    if ( v5 == 1 )
    {
      v9[0] = "After";
      v10 = 259;
    }
    else
    {
      if ( *(_BYTE *)(a2 + 33) == 1 )
      {
        v2 = *(const char **)(a2 + 8);
        v3 = *(const char **)a2;
      }
      else
      {
        v5 = 2;
      }
      v9[2] = v3;
      v9[0] = "After";
      v9[3] = v2;
      LOBYTE(v10) = 3;
      HIBYTE(v10) = v5;
    }
  }
  else
  {
    v10 = 256;
  }
  v6 = *(_QWORD *)(a1 + 32);
  if ( v6 == *(_QWORD *)(a1 + 40) + 48LL || !v6 )
    v7 = 0;
  else
    v7 = v6 - 24;
  return sub_315E1E0(v7, v9);
}
