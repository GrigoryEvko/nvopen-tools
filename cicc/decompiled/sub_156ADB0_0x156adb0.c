// Function: sub_156ADB0
// Address: 0x156adb0
//
__int64 __fastcall sub_156ADB0(__int64 *a1, __int64 a2, __int64 *a3, _BYTE *a4, char a5)
{
  _QWORD *v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rax
  unsigned int v11; // r15d
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v15; // rax
  _QWORD *v16; // r14
  __int64 v17; // rdi
  unsigned __int64 *v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int64 *v25; // r15
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rsi
  _QWORD *v29; // rdx
  __int64 v30; // rsi
  __int64 v32; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v33[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v34; // [rsp+30h] [rbp-60h]
  _BYTE v35[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v36; // [rsp+50h] [rbp-40h]

  v8 = (_QWORD *)a2;
  v9 = *a3;
  v34 = 257;
  v10 = sub_1646BA0(v9, 0);
  if ( v10 != *(_QWORD *)a2 )
  {
    if ( *(_BYTE *)(a2 + 16) > 0x10u )
    {
      v36 = 257;
      v23 = sub_15FDBD0(47, a2, v10, v35, 0);
      v24 = a1[1];
      v8 = (_QWORD *)v23;
      if ( v24 )
      {
        v25 = (unsigned __int64 *)a1[2];
        sub_157E9D0(v24 + 40, v23);
        v26 = v8[3];
        v27 = *v25;
        v8[4] = v25;
        v27 &= 0xFFFFFFFFFFFFFFF8LL;
        v8[3] = v27 | v26 & 7;
        *(_QWORD *)(v27 + 8) = v8 + 3;
        *v25 = *v25 & 7 | (unsigned __int64)(v8 + 3);
      }
      sub_164B780(v8, v33);
      v28 = *a1;
      if ( *a1 )
      {
        v32 = *a1;
        sub_1623A60(&v32, v28, 2);
        v29 = v8 + 6;
        if ( v8[6] )
        {
          sub_161E7C0(v8 + 6);
          v29 = v8 + 6;
        }
        v30 = v32;
        v8[6] = v32;
        if ( v30 )
          sub_1623210(&v32, v30, v29);
      }
    }
    else
    {
      v8 = (_QWORD *)sub_15A46C0(47, a2, v10, 0);
    }
  }
  v11 = 1;
  if ( a5 )
  {
    v12 = *a3;
    v11 = (*(_DWORD *)(v12 + 32) * (unsigned int)sub_1643030(*(_QWORD *)(*a3 + 24))) >> 3;
  }
  if ( a4[16] <= 0x10u && (unsigned __int8)sub_1596070(a4) )
  {
    v36 = 257;
    v15 = sub_1648A60(64, 2);
    v16 = (_QWORD *)v15;
    if ( v15 )
      sub_15F9650(v15, a3, v8, 0, 0);
    v17 = a1[1];
    if ( v17 )
    {
      v18 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v17 + 40, v16);
      v19 = v16[3];
      v20 = *v18;
      v16[4] = v18;
      v20 &= 0xFFFFFFFFFFFFFFF8LL;
      v16[3] = v20 | v19 & 7;
      *(_QWORD *)(v20 + 8) = v16 + 3;
      *v18 = *v18 & 7 | (unsigned __int64)(v16 + 3);
    }
    sub_164B780(v16, v35);
    v21 = *a1;
    if ( *a1 )
    {
      v33[0] = *a1;
      sub_1623A60(v33, v21, 2);
      if ( v16[6] )
        sub_161E7C0(v16 + 6);
      v22 = v33[0];
      v16[6] = v33[0];
      if ( v22 )
        sub_1623210(v33, v22, v16 + 6);
    }
    return sub_15F9450(v16, v11);
  }
  else
  {
    v13 = sub_156A930(a1, a4, *(_QWORD *)(*a3 + 32));
    return sub_15E80D0(a1, a3, v8, v11, v13);
  }
}
