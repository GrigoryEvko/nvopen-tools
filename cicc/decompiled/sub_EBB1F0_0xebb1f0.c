// Function: sub_EBB1F0
// Address: 0xebb1f0
//
__int64 __fastcall sub_EBB1F0(__int64 a1, bool a2)
{
  bool v2; // r13
  char *v5; // rsi
  bool v6; // zf
  unsigned int v7; // r14d
  unsigned __int8 v9; // al
  __int64 v10; // rdi
  __int64 v11; // r15
  bool v12; // al
  void *v13; // rax
  void *v14; // rax
  const char *v15; // [rsp+0h] [rbp-70h] BYREF
  const char *v16; // [rsp+8h] [rbp-68h]
  const char *v17[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v18; // [rsp+30h] [rbp-40h]

  v2 = a2;
  v15 = 0;
  v5 = *(char **)(a1 + 328);
  v16 = 0;
  if ( v5 == *(char **)(a1 + 336) )
  {
    sub_EA9230((char **)(a1 + 320), v5, (_QWORD *)(a1 + 308));
  }
  else
  {
    if ( v5 )
    {
      *(_QWORD *)v5 = *(_QWORD *)(a1 + 308);
      v5 = *(char **)(a1 + 328);
    }
    *(_QWORD *)(a1 + 328) = v5 + 8;
  }
  v6 = *(_BYTE *)(a1 + 313) == 0;
  *(_DWORD *)(a1 + 308) = 1;
  if ( v6 )
  {
    v18 = 259;
    v17[0] = "expected identifier after '.ifdef'";
    v9 = sub_EB61F0(a1, (__int64 *)&v15);
    if ( (unsigned __int8)sub_ECE0A0(a1, v9, v17) )
      return 1;
    v7 = sub_ECE000(a1);
    if ( (_BYTE)v7 )
    {
      return 1;
    }
    else
    {
      v10 = *(_QWORD *)(a1 + 224);
      v18 = 261;
      v17[0] = v15;
      v17[1] = v16;
      v11 = sub_E65280(v10, v17);
      if ( a2 )
      {
        v12 = 0;
        if ( v11 )
        {
          if ( *(_QWORD *)v11 )
          {
            v12 = a2;
            v2 = 0;
          }
          else if ( (*(_BYTE *)(v11 + 9) & 0x70) == 0x20 && *(char *)(v11 + 8) >= 0 )
          {
            v14 = sub_E807D0(*(_QWORD *)(v11 + 24));
            *(_QWORD *)v11 = v14;
            v2 = v14 == 0;
            v12 = v14 != 0;
          }
        }
      }
      else
      {
        v12 = 1;
        if ( v11 )
        {
          if ( *(_QWORD *)v11 )
          {
            v12 = 0;
            v2 = 1;
          }
          else if ( (*(_BYTE *)(v11 + 9) & 0x70) == 0x20 && *(char *)(v11 + 8) >= 0 )
          {
            v13 = sub_E807D0(*(_QWORD *)(v11 + 24));
            *(_QWORD *)v11 = v13;
            v2 = v13 != 0;
            v12 = v13 == 0;
          }
        }
      }
      *(_BYTE *)(a1 + 312) = v12;
      *(_BYTE *)(a1 + 313) = v2;
    }
  }
  else
  {
    v7 = 0;
    sub_EB4E00(a1);
  }
  return v7;
}
