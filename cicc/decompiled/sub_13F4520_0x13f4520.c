// Function: sub_13F4520
// Address: 0x13f4520
//
void __fastcall sub_13F4520(_QWORD *a1, _BYTE *a2)
{
  __int64 v3; // rsi
  __int64 v4; // r13
  unsigned int v5; // r15d
  _BYTE *v6; // rax
  _BYTE *v7; // rax
  const char *v8; // [rsp+0h] [rbp-80h] BYREF
  _BYTE *v9; // [rsp+8h] [rbp-78h]
  _BYTE *v10; // [rsp+10h] [rbp-70h]
  __int64 v11; // [rsp+18h] [rbp-68h]
  int v12; // [rsp+20h] [rbp-60h]
  _BYTE v13[88]; // [rsp+28h] [rbp-58h] BYREF

  v3 = *((_QWORD *)a2 - 3);
  v8 = 0;
  v9 = v13;
  v10 = v13;
  v11 = 4;
  v12 = 0;
  v4 = sub_13F3D10(a1, v3, 0, (__int64)&v8);
  if ( v10 != v9 )
    _libc_free((unsigned __int64)v10);
  if ( *(_BYTE *)(v4 + 16) == 13 )
  {
    v5 = *(_DWORD *)(v4 + 32);
    if ( v5 <= 0x40 )
    {
      if ( (unsigned __int64)(*(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8) > *(_QWORD *)(v4 + 24) )
        return;
LABEL_9:
      v8 = "Undefined result: Shift count out of range";
      LOWORD(v10) = 259;
      sub_16E2CE0(&v8, a1 + 30);
      v6 = (_BYTE *)a1[33];
      if ( (unsigned __int64)v6 >= a1[32] )
      {
        sub_16E7DE0(a1 + 30, 10);
      }
      else
      {
        a1[33] = v6 + 1;
        *v6 = 10;
      }
      if ( a2[16] <= 0x17u )
      {
        sub_15537D0(a2, a1 + 30, 1);
        v7 = (_BYTE *)a1[33];
        if ( (unsigned __int64)v7 < a1[32] )
          goto LABEL_13;
      }
      else
      {
        sub_155C2B0(a2, a1 + 30, 0);
        v7 = (_BYTE *)a1[33];
        if ( (unsigned __int64)v7 < a1[32] )
        {
LABEL_13:
          a1[33] = v7 + 1;
          *v7 = 10;
          return;
        }
      }
      sub_16E7DE0(a1 + 30, 10);
      return;
    }
    if ( v5 - (unsigned int)sub_16A57B0(v4 + 24) > 0x40
      || (unsigned __int64)(*(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8) <= **(_QWORD **)(v4 + 24) )
    {
      goto LABEL_9;
    }
  }
}
