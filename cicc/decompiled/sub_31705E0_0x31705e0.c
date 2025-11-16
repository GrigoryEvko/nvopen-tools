// Function: sub_31705E0
// Address: 0x31705e0
//
__int64 __fastcall sub_31705E0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v6; // rdi
  __int64 v7; // rsi
  _QWORD *v8; // rdx
  unsigned __int64 v9; // rax
  _QWORD *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 *v15; // rbx
  _QWORD *v16; // r15
  unsigned __int64 v17; // rax
  int v18; // edx
  _QWORD *v19; // rdi
  _QWORD *v20; // rax
  __int64 *v21; // rax
  unsigned __int16 v22; // bx
  _QWORD *v23; // rax
  __int64 v24; // r14
  unsigned __int16 v25; // bx
  _QWORD *v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // [rsp+8h] [rbp-88h]
  __int64 v30; // [rsp+10h] [rbp-80h]
  __int64 v31; // [rsp+10h] [rbp-80h]
  __int64 v32; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int16 v34; // [rsp+28h] [rbp-68h]
  __int64 v35; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int16 v36; // [rsp+38h] [rbp-58h]
  __int16 v37; // [rsp+50h] [rbp-40h]

  if ( *(_BYTE *)a2 == 22 )
  {
    v3 = a1[39];
    if ( *(_BYTE *)v3 <= 0x1Cu )
    {
      v3 = *(_QWORD *)(*(_QWORD *)(v3 + 24) + 80LL);
      if ( !v3 )
        BUG();
    }
    v4 = *(_QWORD *)(v3 + 32);
    sub_B2D580(*(_QWORD *)(a2 + 24), *(_DWORD *)(a2 + 32), 89);
    return v4;
  }
  if ( *(_BYTE *)a2 == 85 )
  {
    v11 = *(_QWORD *)(a2 - 32);
    if ( v11 )
    {
      if ( !*(_BYTE *)v11
        && *(_QWORD *)(v11 + 24) == *(_QWORD *)(a2 + 80)
        && (*(_BYTE *)(v11 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v11 + 36) - 60) <= 2 )
      {
        v12 = sub_AA56F0(*(_QWORD *)(a2 + 40));
        return sub_AA4FF0(v12);
      }
    }
  }
  if ( (unsigned __int8)sub_B19DB0(a3, *a1, a2) )
  {
    v6 = *(_QWORD *)(a2 + 40);
    if ( *(_BYTE *)a2 == 34 )
    {
      v7 = *(_QWORD *)(a2 - 96);
      v37 = 257;
      v8 = (_QWORD *)(sub_F41C30(v6, v7, 0, 0, 0, (void **)&v35) + 48);
      v9 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v9 == v8 )
      {
        v10 = 0;
      }
      else
      {
        if ( !v9 )
          BUG();
        v10 = (_QWORD *)(v9 - 24);
        if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 >= 0xB )
          v10 = 0;
      }
      return (__int64)(v10 + 3);
    }
    v14 = (__int64 *)(v6 + 48);
    if ( *(_BYTE *)a2 == 84 )
    {
      v15 = (__int64 *)(*(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL);
      if ( v15 == v14 )
        goto LABEL_46;
      if ( !v15 )
        BUG();
      if ( (unsigned int)*((unsigned __int8 *)v15 - 24) - 30 > 0xA )
LABEL_46:
        BUG();
      if ( *((_BYTE *)v15 - 24) != 39 )
        return sub_AA5190(v6);
      v16 = (_QWORD *)v15[2];
      v37 = 257;
      v32 = sub_AA8550(v16, v15, 0, (__int64)&v35, 0);
      v17 = v16[6] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v17 == v16 + 6 )
      {
        v19 = 0;
      }
      else
      {
        if ( !v17 )
          BUG();
        v18 = *(unsigned __int8 *)(v17 - 24);
        v19 = 0;
        v20 = (_QWORD *)(v17 - 24);
        if ( (unsigned int)(v18 - 30) < 0xB )
          v19 = v20;
      }
      sub_B43D60(v19);
      sub_B43C20((__int64)&v33, (__int64)v16);
      v37 = 257;
      v21 = (__int64 *)*(v15 - 4);
      v22 = v34;
      v29 = *v21;
      v30 = v33;
      v23 = sub_BD2C40(72, 1u);
      v24 = (__int64)v23;
      if ( v23 )
        sub_B4C840((__int64)v23, 51, v29, 0, 0, 1u, (__int64)&v35, v30, v22);
      sub_B43C20((__int64)&v35, (__int64)v16);
      v25 = v36;
      v31 = v35;
      v26 = sub_BD2C40(72, 2 - (unsigned int)(v32 == 0));
      v10 = v26;
      if ( v26 )
        sub_B4BF70((__int64)v26, v24, v32, (2 - (v32 == 0)) & 0x1FFFFFFF, v31, v25);
      return (__int64)(v10 + 3);
    }
    v27 = *(__int64 **)(a2 + 32);
    if ( !v27 || v27 == v14 )
      v28 = 0;
    else
      v28 = v27 - 3;
    return (__int64)(v28 + 3);
  }
  else
  {
    v13 = a1[39];
    if ( *(_BYTE *)v13 <= 0x1Cu )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(v13 + 24) + 80LL);
      if ( !v13 )
        BUG();
    }
    return *(_QWORD *)(v13 + 32);
  }
}
