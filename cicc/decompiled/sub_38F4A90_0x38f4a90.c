// Function: sub_38F4A90
// Address: 0x38f4a90
//
__int64 __fastcall sub_38F4A90(__int64 a1, bool a2, __int64 a3, unsigned int a4)
{
  char *v6; // rsi
  bool v7; // zf
  unsigned int v8; // r14d
  unsigned __int8 v10; // al
  __int64 v11; // rdi
  __int64 v12; // r15
  bool v13; // al
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  bool v16; // [rsp+Fh] [rbp-81h]
  __int64 v17[2]; // [rsp+10h] [rbp-80h] BYREF
  const char *v18; // [rsp+20h] [rbp-70h] BYREF
  char v19; // [rsp+30h] [rbp-60h]
  char v20; // [rsp+31h] [rbp-5Fh]
  _QWORD v21[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v22; // [rsp+50h] [rbp-40h]

  v16 = a2;
  v6 = *(char **)(a1 + 400);
  v17[0] = 0;
  v17[1] = 0;
  if ( v6 == *(char **)(a1 + 408) )
  {
    sub_38E9AD0((unsigned __int64 *)(a1 + 392), v6, (_QWORD *)(a1 + 380));
  }
  else
  {
    if ( v6 )
    {
      *(_QWORD *)v6 = *(_QWORD *)(a1 + 380);
      v6 = *(char **)(a1 + 400);
    }
    *(_QWORD *)(a1 + 400) = v6 + 8;
  }
  v7 = *(_BYTE *)(a1 + 385) == 0;
  *(_DWORD *)(a1 + 380) = 1;
  if ( v7 )
  {
    v22 = 259;
    v21[0] = "expected identifier after '.ifdef'";
    v10 = sub_38F0EE0(a1, v17, a3, a4);
    if ( (unsigned __int8)sub_3909CB0(a1, v10, v21) )
      return 1;
    v20 = 1;
    v19 = 3;
    v18 = "unexpected token in '.ifdef'";
    v8 = sub_3909E20(a1, 9, &v18);
    if ( (_BYTE)v8 )
    {
      return 1;
    }
    else
    {
      v11 = *(_QWORD *)(a1 + 320);
      v21[0] = v17;
      v22 = 261;
      v12 = sub_38BD730(v11, (__int64)v21);
      if ( a2 )
      {
        v13 = 0;
        if ( v12 )
        {
          if ( (*(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v16 = 0;
            v13 = a2;
          }
          else if ( (*(_BYTE *)(v12 + 9) & 0xC) == 8 )
          {
            *(_BYTE *)(v12 + 8) |= 4u;
            v14 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v12 + 24));
            *(_QWORD *)v12 = v14 | *(_QWORD *)v12 & 7LL;
            v13 = v14 != 0;
            v16 = v14 == 0;
          }
        }
      }
      else
      {
        v13 = 1;
        if ( v12 )
        {
          if ( (*(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v16 = 1;
            v13 = 0;
          }
          else if ( (*(_BYTE *)(v12 + 9) & 0xC) == 8 )
          {
            *(_BYTE *)(v12 + 8) |= 4u;
            v15 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v12 + 24));
            *(_QWORD *)v12 = v15 | *(_QWORD *)v12 & 7LL;
            v13 = v15 == 0;
            v16 = v15 != 0;
          }
        }
      }
      *(_BYTE *)(a1 + 384) = v13;
      *(_BYTE *)(a1 + 385) = v16;
    }
  }
  else
  {
    v8 = 0;
    sub_38F0630(a1);
  }
  return v8;
}
