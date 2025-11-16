// Function: sub_1BC4A30
// Address: 0x1bc4a30
//
__int64 __fastcall sub_1BC4A30(_QWORD *a1, unsigned __int64 a2, __int64 a3, __int64 **a4)
{
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r10
  __int64 v14; // rax
  _QWORD *v15; // rbx
  bool v16; // zf
  __int64 v18; // rdx
  __int64 v19; // rsi
  int v20; // edi
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 *v23; // r13
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  int v29; // edx
  int v30; // r11d
  unsigned __int64 v31; // [rsp+8h] [rbp-78h] BYREF
  unsigned __int8 *v32; // [rsp+18h] [rbp-68h] BYREF
  __int64 v33; // [rsp+20h] [rbp-60h] BYREF
  __int16 v34; // [rsp+30h] [rbp-50h]
  char v35[16]; // [rsp+40h] [rbp-40h] BYREF
  __int16 v36; // [rsp+50h] [rbp-30h]

  v6 = (_QWORD *)*a1;
  v31 = a2;
  v7 = *((unsigned int *)v6 + 374);
  if ( !(_DWORD)v7 )
    return a3;
  v10 = v6[185];
  v11 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (__int64 *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( a2 != *v12 )
  {
    v29 = 1;
    while ( v13 != -8 )
    {
      v30 = v29 + 1;
      v11 = (v7 - 1) & (v29 + v11);
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( a2 == *v12 )
        goto LABEL_3;
      v29 = v30;
    }
    return a3;
  }
LABEL_3:
  if ( v12 == (__int64 *)(v10 + 16 * v7) )
    return a3;
  v14 = sub_1BC4770((__int64)(v6 + 184), &v31);
  v15 = (_QWORD *)*a1;
  v16 = *(_BYTE *)(v14 + 8) == 0;
  v34 = 257;
  if ( !v16 )
  {
    if ( a4 != *(__int64 ***)a3 )
    {
      if ( *(_BYTE *)(a3 + 16) <= 0x10u )
        return sub_15A46C0(38, (__int64 ***)a3, a4, 0);
      v19 = a3;
      v36 = 257;
      v20 = 38;
      v18 = (__int64)a4;
LABEL_13:
      v21 = sub_15FDBD0(v20, v19, v18, (__int64)v35, 0);
      v22 = v15[176];
      a3 = v21;
      if ( v22 )
      {
        v23 = (__int64 *)v15[177];
        sub_157E9D0(v22 + 40, v21);
        v24 = *(_QWORD *)(a3 + 24);
        v25 = *v23;
        *(_QWORD *)(a3 + 32) = v23;
        v25 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a3 + 24) = v25 | v24 & 7;
        *(_QWORD *)(v25 + 8) = a3 + 24;
        *v23 = *v23 & 7 | (a3 + 24);
      }
      sub_164B780(a3, &v33);
      v26 = v15[175];
      if ( v26 )
      {
        v32 = (unsigned __int8 *)v15[175];
        sub_1623A60((__int64)&v32, v26, 2);
        v27 = *(_QWORD *)(a3 + 48);
        if ( v27 )
          sub_161E7C0(a3 + 48, v27);
        v28 = v32;
        *(_QWORD *)(a3 + 48) = v32;
        if ( v28 )
          sub_1623210((__int64)&v32, v28, a3 + 48);
      }
    }
    return a3;
  }
  if ( a4 == *(__int64 ***)a3 )
    return a3;
  if ( *(_BYTE *)(a3 + 16) > 0x10u )
  {
    v18 = (__int64)a4;
    v36 = 257;
    v19 = a3;
    v20 = 37;
    goto LABEL_13;
  }
  return sub_15A46C0(37, (__int64 ***)a3, a4, 0);
}
