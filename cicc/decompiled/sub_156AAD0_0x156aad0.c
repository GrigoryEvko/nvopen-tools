// Function: sub_156AAD0
// Address: 0x156aad0
//
__int64 __fastcall sub_156AAD0(__int64 *a1, __int64 a2, __int64 *a3, _BYTE *a4, char a5)
{
  _QWORD *v5; // r15
  __int64 v9; // rdi
  __int64 v10; // rax
  unsigned int v11; // r14d
  __int64 v12; // rax
  _QWORD *v13; // r13
  __int64 v15; // rdi
  unsigned __int64 *v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned __int64 *v23; // r14
  __int64 v24; // rax
  unsigned __int64 v25; // rcx
  __int64 v26; // rsi
  _QWORD *v27; // rdx
  __int64 v28; // rsi
  __int64 v30; // [rsp+18h] [rbp-88h]
  __int64 v31; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v32[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v33; // [rsp+40h] [rbp-60h]
  _BYTE v34[16]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v35; // [rsp+60h] [rbp-40h]

  v5 = (_QWORD *)a2;
  v9 = *a3;
  v33 = 257;
  v10 = sub_1646BA0(v9, 0);
  if ( v10 != *(_QWORD *)a2 )
  {
    if ( *(_BYTE *)(a2 + 16) > 0x10u )
    {
      v35 = 257;
      v21 = sub_15FDBD0(47, a2, v10, v34, 0);
      v22 = a1[1];
      v5 = (_QWORD *)v21;
      if ( v22 )
      {
        v23 = (unsigned __int64 *)a1[2];
        sub_157E9D0(v22 + 40, v21);
        v24 = v5[3];
        v25 = *v23;
        v5[4] = v23;
        v25 &= 0xFFFFFFFFFFFFFFF8LL;
        v5[3] = v25 | v24 & 7;
        *(_QWORD *)(v25 + 8) = v5 + 3;
        *v23 = *v23 & 7 | (unsigned __int64)(v5 + 3);
      }
      sub_164B780(v5, v32);
      v26 = *a1;
      if ( *a1 )
      {
        v31 = *a1;
        sub_1623A60(&v31, v26, 2);
        v27 = v5 + 6;
        if ( v5[6] )
        {
          sub_161E7C0(v5 + 6);
          v27 = v5 + 6;
        }
        v28 = v31;
        v5[6] = v31;
        if ( v28 )
          sub_1623210(&v31, v28, v27);
      }
    }
    else
    {
      v5 = (_QWORD *)sub_15A46C0(47, a2, v10, 0);
    }
  }
  v11 = 1;
  if ( a5 )
  {
    v30 = *a3;
    v11 = (*(_DWORD *)(v30 + 32) * (unsigned int)sub_1643030(*(_QWORD *)(*a3 + 24))) >> 3;
  }
  if ( a4[16] <= 0x10u && (unsigned __int8)sub_1596070(a4) )
  {
    v35 = 257;
    v13 = (_QWORD *)sub_1648A60(64, 1);
    if ( v13 )
      sub_15F9210(v13, *(_QWORD *)(*v5 + 24LL), v5, 0, 0, 0);
    v15 = a1[1];
    if ( v15 )
    {
      v16 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v15 + 40, v13);
      v17 = v13[3];
      v18 = *v16;
      v13[4] = v16;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      v13[3] = v18 | v17 & 7;
      *(_QWORD *)(v18 + 8) = v13 + 3;
      *v16 = *v16 & 7 | (unsigned __int64)(v13 + 3);
    }
    sub_164B780(v13, v34);
    v19 = *a1;
    if ( *a1 )
    {
      v32[0] = *a1;
      sub_1623A60(v32, v19, 2);
      if ( v13[6] )
        sub_161E7C0(v13 + 6);
      v20 = v32[0];
      v13[6] = v32[0];
      if ( v20 )
        sub_1623210(v32, v20, v13 + 6);
    }
    sub_15F8F50(v13, v11);
  }
  else
  {
    v12 = sub_156A930(a1, a4, *(_QWORD *)(*a3 + 32));
    v35 = 257;
    return sub_15E8010(a1, v5, v11, v12, a3, v34);
  }
  return (__int64)v13;
}
