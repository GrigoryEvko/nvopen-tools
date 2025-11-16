// Function: sub_1289860
// Address: 0x1289860
//
__int64 __fastcall sub_1289860(_QWORD *a1, __int64 a2, _BYTE *a3)
{
  _QWORD *v3; // r12
  __int64 v4; // rax
  bool v6; // cc
  __int64 v7; // rax
  unsigned __int64 *v8; // r13
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // [rsp+8h] [rbp-68h] BYREF
  const char *v14; // [rsp+10h] [rbp-60h] BYREF
  char v15; // [rsp+20h] [rbp-50h]
  char v16; // [rsp+21h] [rbp-4Fh]
  char v17[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v18; // [rsp+40h] [rbp-30h]

  v3 = a3;
  v4 = *(_QWORD *)a3;
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 15
    && *(_BYTE *)(a2 + 8) == 15
    && a2 != v4
    && *(_BYTE *)(*(_QWORD *)(a2 + 24) + 8LL) == 14
    && *(_BYTE *)(*(_QWORD *)(v4 + 24) + 8LL) == 14 )
  {
    v6 = a3[16] <= 0x10u;
    v16 = 1;
    v14 = "ptrcast";
    v15 = 3;
    if ( v6 )
    {
      return sub_15A46C0(47, a3, a2, 0);
    }
    else
    {
      v18 = 257;
      v3 = (_QWORD *)sub_15FDBD0(47, a3, a2, v17, 0);
      v7 = a1[7];
      if ( v7 )
      {
        v8 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v7 + 40, v3);
        v9 = v3[3];
        v10 = *v8;
        v3[4] = v8;
        v10 &= 0xFFFFFFFFFFFFFFF8LL;
        v3[3] = v10 | v9 & 7;
        *(_QWORD *)(v10 + 8) = v3 + 3;
        *v8 = *v8 & 7 | (unsigned __int64)(v3 + 3);
      }
      sub_164B780(v3, &v14);
      v11 = a1[6];
      if ( v11 )
      {
        v13 = a1[6];
        sub_1623A60(&v13, v11, 2);
        if ( v3[6] )
          sub_161E7C0(v3 + 6);
        v12 = v13;
        v3[6] = v13;
        if ( v12 )
          sub_1623210(&v13, v12, v3 + 6);
      }
    }
  }
  return (__int64)v3;
}
