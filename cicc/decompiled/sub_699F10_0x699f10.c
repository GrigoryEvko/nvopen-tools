// Function: sub_699F10
// Address: 0x699f10
//
__int64 __fastcall sub_699F10(__int64 a1, _BYTE *a2, char a3)
{
  unsigned int v5; // r12d
  unsigned int v7; // r15d
  _BYTE *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  _BYTE *v15; // r8
  _BOOL4 v16; // r12d
  __int64 v17; // [rsp-10h] [rbp-110h]
  __int64 v18; // [rsp-8h] [rbp-108h]
  int v19; // [rsp+8h] [rbp-F8h]
  _BYTE v20[4]; // [rsp+1Ch] [rbp-E4h] BYREF
  __int64 v21; // [rsp+20h] [rbp-E0h] BYREF
  char v22[8]; // [rsp+28h] [rbp-D8h] BYREF
  _BYTE v23[208]; // [rsp+30h] [rbp-D0h] BYREF

  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned int)sub_8D23B0(a1) )
      sub_8AE000(a1);
    if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(a2) )
      sub_8AE000(a2);
  }
  v5 = sub_8D2600(a2);
  if ( v5 )
    return sub_8D2600(a1);
  if ( !(unsigned int)sub_8D2600(a1)
    && !(unsigned int)sub_8D3410(a2)
    && !(unsigned int)sub_8D2310(a2)
    && !(unsigned int)sub_8D23B0(a1)
    && !(unsigned int)sub_8D23B0(a2) )
  {
    v7 = sub_8D5830(a2);
    if ( !v7 )
    {
      sub_6E1DD0(&v21);
      v8 = v23;
      sub_6E1E00(5, v23, 0, 1);
      *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x80u;
      v9 = sub_68B9A0(a1);
      v10 = v9;
      if ( v9 )
      {
        v11 = *(_QWORD *)(v9 + 24);
        v19 = v11 + 8;
        v12 = sub_732490(a2, v20);
        if ( (_DWORD)v12 )
        {
          v8 = a2;
          sub_8470D0(v19, (_DWORD)a2, 0, 262274, 0, 0, (__int64)v22);
          v15 = v20;
        }
        else
        {
          v18 = v12;
          v8 = a2;
          sub_843C40(v19, (_DWORD)a2, 0, 0, 1, 262274, 0);
          v13 = v17;
          v14 = v18;
        }
        v16 = (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0;
        if ( a3 == 108 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 && *(_BYTE *)(v11 + 24) == 1 )
        {
          if ( (unsigned int)sub_731B40(*(_QWORD *)(v11 + 152), v8, v13, v14, v15) )
            v16 = 0;
        }
        a1 = v10;
        v7 = v16;
        sub_6E1990(v10);
      }
      sub_6E2B30(a1, v8);
      v5 = v7;
      sub_6E1DF0(v21);
    }
  }
  return v5;
}
