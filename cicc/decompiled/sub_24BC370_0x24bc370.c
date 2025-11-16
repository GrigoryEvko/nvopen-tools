// Function: sub_24BC370
// Address: 0x24bc370
//
char __fastcall sub_24BC370(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // r14
  _BYTE *v4; // rax
  __int64 v5; // rsi
  char *v6; // rsi
  unsigned int v8; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = (__int64 *)a1[4];
  if ( !(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 23) && (LOBYTE(v4) = sub_B49560(a2, 23), !(_BYTE)v4)
    || (LOBYTE(v4) = sub_A73ED0((_QWORD *)(a2 + 72), 4), (_BYTE)v4)
    || (LOBYTE(v4) = sub_B49560(a2, 4), (_BYTE)v4) )
  {
    v5 = *(_QWORD *)(a2 - 32);
    if ( v5 )
    {
      if ( !*(_BYTE *)v5 )
      {
        v4 = *(_BYTE **)(a2 + 80);
        if ( *(_BYTE **)(v5 + 24) == v4 )
        {
          LOBYTE(v4) = sub_981210(*v3, v5, &v8);
          if ( (_BYTE)v4 )
          {
            LOBYTE(v4) = v8;
            if ( v8 == 357 || v8 == 186 )
            {
              v4 = *(_BYTE **)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
              if ( *v4 != 17 )
              {
                v9[0] = a2;
                v6 = (char *)a1[7];
                if ( v6 == (char *)a1[8] )
                {
                  LOBYTE(v4) = sub_24BBE90(a1 + 6, v6, v9);
                }
                else
                {
                  if ( v6 )
                  {
                    *(_QWORD *)v6 = a2;
                    v6 = (char *)a1[7];
                  }
                  a1[7] = v6 + 8;
                }
              }
            }
          }
        }
      }
    }
  }
  return (char)v4;
}
