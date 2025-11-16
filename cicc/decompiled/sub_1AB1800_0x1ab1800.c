// Function: sub_1AB1800
// Address: 0x1ab1800
//
__int64 __fastcall sub_1AB1800(__int64 a1, __int64 *a2)
{
  __int64 v2; // r12
  __int64 v4; // rax
  unsigned int v5; // esi
  _QWORD *v6; // rdi
  __int64 **v7; // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rsi
  unsigned __int8 *v16; // rsi
  unsigned __int8 *v17; // [rsp+8h] [rbp-68h] BYREF
  const char *v18; // [rsp+10h] [rbp-60h] BYREF
  char v19; // [rsp+20h] [rbp-50h]
  char v20; // [rsp+21h] [rbp-4Fh]
  char v21[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v22; // [rsp+40h] [rbp-30h]

  v2 = a1;
  v4 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
    v4 = **(_QWORD **)(v4 + 16);
  v5 = *(_DWORD *)(v4 + 8);
  v6 = (_QWORD *)a2[3];
  v20 = 1;
  v18 = "cstr";
  v19 = 3;
  v7 = (__int64 **)sub_16471D0(v6, v5 >> 8);
  if ( v7 != *(__int64 ***)v2 )
  {
    if ( *(_BYTE *)(v2 + 16) > 0x10u )
    {
      v22 = 257;
      v9 = sub_15FDBD0(47, v2, (__int64)v7, (__int64)v21, 0);
      v10 = a2[1];
      v2 = v9;
      if ( v10 )
      {
        v11 = (__int64 *)a2[2];
        sub_157E9D0(v10 + 40, v9);
        v12 = *(_QWORD *)(v2 + 24);
        v13 = *v11;
        *(_QWORD *)(v2 + 32) = v11;
        v13 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v2 + 24) = v13 | v12 & 7;
        *(_QWORD *)(v13 + 8) = v2 + 24;
        *v11 = *v11 & 7 | (v2 + 24);
      }
      sub_164B780(v2, (__int64 *)&v18);
      v14 = *a2;
      if ( *a2 )
      {
        v17 = (unsigned __int8 *)*a2;
        sub_1623A60((__int64)&v17, v14, 2);
        v15 = *(_QWORD *)(v2 + 48);
        if ( v15 )
          sub_161E7C0(v2 + 48, v15);
        v16 = v17;
        *(_QWORD *)(v2 + 48) = v17;
        if ( v16 )
          sub_1623210((__int64)&v17, v16, v2 + 48);
      }
    }
    else
    {
      return sub_15A46C0(47, (__int64 ***)v2, v7, 0);
    }
  }
  return v2;
}
