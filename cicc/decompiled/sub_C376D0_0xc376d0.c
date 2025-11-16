// Function: sub_C376D0
// Address: 0xc376d0
//
__int64 __fastcall sub_C376D0(__int64 a1, __int64 a2, char a3)
{
  char v4; // al
  int v5; // esi
  signed int v6; // r14d
  unsigned int v7; // r14d
  int v8; // eax
  __int64 v9; // rdx
  bool v10; // bl
  char v11; // r13
  int v12; // eax
  unsigned __int8 v13; // bl
  _QWORD v15[10]; // [rsp+0h] [rbp-50h] BYREF

  v4 = *(_BYTE *)(a2 + 20) ^ *(_BYTE *)(a1 + 20);
  v5 = *(_DWORD *)(a2 + 16);
  v6 = *(_DWORD *)(a1 + 16) - v5;
  if ( ((v4 & 8) != 0) != a3 )
  {
    if ( v6 < 0 )
    {
      if ( !*(_BYTE *)(*(_QWORD *)a1 + 25LL) )
        BUG();
      sub_C33EB0(v15, (__int64 *)a2);
      v7 = sub_C342A0(a1, ~v6);
      sub_C34340((__int64)v15, 1);
      v8 = sub_C37580(a1, (__int64)v15);
      if ( !v8 )
      {
        if ( v7 )
        {
          v9 = 1;
          if ( v7 == 1 )
          {
            v7 = 3;
          }
          else if ( v7 == 3 )
          {
            v7 = 1;
          }
        }
        else
        {
          v9 = 0;
        }
        goto LABEL_9;
      }
      v13 = 0;
      if ( v8 != 2 )
      {
        sub_C33EE0(a1);
        goto LABEL_21;
      }
LABEL_18:
      sub_C33F90(a1, (__int64)v15, v13);
      goto LABEL_21;
    }
    sub_C33EB0(v15, (__int64 *)a2);
    if ( v6 )
    {
      v11 = 1;
      v7 = sub_C342A0((__int64)v15, v6 - 1);
      sub_C34340(a1, 1);
      v10 = v7 != 0;
      v12 = sub_C37580(a1, (__int64)v15);
      if ( v12 )
        goto LABEL_13;
    }
    else
    {
      v10 = 0;
      v11 = 0;
      v7 = 0;
      v12 = sub_C37580(a1, (__int64)v15);
      if ( v12 )
      {
LABEL_13:
        if ( v12 != 2 )
        {
          sub_C33EE0(a1);
          if ( !v11 || !v10 )
            goto LABEL_21;
          goto LABEL_10;
        }
        v13 = v11 & v10;
        if ( v13 )
        {
          if ( v7 == 1 )
          {
            v7 = 3;
          }
          else if ( v7 == 3 )
          {
            v7 = 1;
          }
        }
        goto LABEL_18;
      }
    }
    v9 = ((unsigned __int8)v11 ^ 1u) & v10;
LABEL_9:
    sub_C33F90((__int64)v15, a1, v9);
    sub_C33DD0(a1, (__int64)v15);
LABEL_10:
    *(_BYTE *)(a1 + 20) = ~*(_BYTE *)(a1 + 20) & 8 | *(_BYTE *)(a1 + 20) & 0xF7;
    goto LABEL_21;
  }
  if ( v6 <= 0 )
  {
    v7 = sub_C342A0(a1, v5 - *(_DWORD *)(a1 + 16));
    sub_C33F40(a1, a2);
    return v7;
  }
  sub_C33EB0(v15, (__int64 *)a2);
  v7 = sub_C342A0((__int64)v15, v6);
  sub_C33F40(a1, (__int64)v15);
LABEL_21:
  sub_C338F0((__int64)v15);
  return v7;
}
