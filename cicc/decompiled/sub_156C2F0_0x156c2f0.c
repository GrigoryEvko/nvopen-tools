// Function: sub_156C2F0
// Address: 0x156c2f0
//
__int64 __fastcall sub_156C2F0(__int64 *a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // r14
  __int64 v5; // r12
  unsigned int v6; // ebx
  __int64 v8; // rax
  __int64 v9; // rax
  _DWORD *v10; // rcx
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // [rsp+8h] [rbp-78h] BYREF
  _BYTE v24[16]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v25; // [rsp+20h] [rbp-60h]
  _DWORD v26[4]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v27; // [rsp+40h] [rbp-40h]

  v3 = a2;
  v5 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
  v6 = v5;
  if ( a3 && (a3[16] > 0x10u || !(unsigned __int8)sub_1596070(a3)) )
  {
    v27 = 257;
    v8 = sub_156A930(a1, a3, v5);
    v3 = sub_1281C00(a1, a2, v8, (__int64)v26);
  }
  if ( (unsigned int)v5 <= 7 )
  {
    if ( (_DWORD)v5 )
    {
      v9 = 0;
      do
      {
        v26[v9] = v9;
        ++v9;
      }
      while ( (_DWORD)v5 != (_DWORD)v9 );
    }
    v10 = &v26[(unsigned int)v5];
    do
    {
      v11 = v5;
      LODWORD(v5) = v5 + 1;
      *v10++ = v6 + v11 % v6;
    }
    while ( (_DWORD)v5 != 8 );
    v25 = 257;
    v12 = sub_15A06D0(*(_QWORD *)v3);
    v3 = sub_156A7D0(a1, v3, v12, (__int64)v26, 8, (__int64)v24);
  }
  v13 = a1[3];
  v25 = 257;
  v14 = sub_1644C60(v13, (unsigned int)v5);
  if ( v14 != *(_QWORD *)v3 )
  {
    if ( *(_BYTE *)(v3 + 16) > 0x10u )
    {
      v27 = 257;
      v16 = sub_15FDBD0(47, v3, v14, v26, 0);
      v17 = a1[1];
      v3 = v16;
      if ( v17 )
      {
        v18 = (__int64 *)a1[2];
        sub_157E9D0(v17 + 40, v16);
        v19 = *(_QWORD *)(v3 + 24);
        v20 = *v18;
        *(_QWORD *)(v3 + 32) = v18;
        v20 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v3 + 24) = v20 | v19 & 7;
        *(_QWORD *)(v20 + 8) = v3 + 24;
        *v18 = *v18 & 7 | (v3 + 24);
      }
      sub_164B780(v3, v24);
      v21 = *a1;
      if ( *a1 )
      {
        v23 = *a1;
        sub_1623A60(&v23, v21, 2);
        if ( *(_QWORD *)(v3 + 48) )
          sub_161E7C0(v3 + 48);
        v22 = v23;
        *(_QWORD *)(v3 + 48) = v23;
        if ( v22 )
          sub_1623210(&v23, v22, v3 + 48);
      }
    }
    else
    {
      return sub_15A46C0(47, v3, v14, 0);
    }
  }
  return v3;
}
