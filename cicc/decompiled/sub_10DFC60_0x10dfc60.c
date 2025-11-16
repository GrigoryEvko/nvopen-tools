// Function: sub_10DFC60
// Address: 0x10dfc60
//
__int64 __fastcall sub_10DFC60(__int64 a1, __int64 *a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rax
  __int64 v4; // r12
  unsigned int v5; // r15d
  unsigned __int64 v6; // r12
  unsigned __int16 v7; // ax
  __int64 *v8; // r8
  unsigned __int8 v9; // r15
  __int64 v10; // rax
  unsigned __int16 v11; // ax
  unsigned __int8 v12; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 *v23; // rax
  __int64 *v24; // rax
  __int64 v25; // r12
  __int64 *v26; // rax
  unsigned int v27; // [rsp+4h] [rbp-6Ch]
  unsigned int v28; // [rsp+4h] [rbp-6Ch]
  unsigned int v29; // [rsp+8h] [rbp-68h]
  __int64 *v30; // [rsp+8h] [rbp-68h]
  __int64 *v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v33; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v34; // [rsp+28h] [rbp-48h]
  unsigned __int8 v35; // [rsp+30h] [rbp-40h]

  sub_D5CDD0((__int64)&v33, a1, a2, sub_10DF250, (__int64)&v32);
  v2 = v35;
  if ( v35 )
  {
    v29 = v34;
    if ( v34 > 0x40 )
    {
      if ( v29 - (unsigned int)sub_C444A0((__int64)&v33) <= 0x40 && !*v33 )
        goto LABEL_4;
    }
    else if ( !v33 )
    {
LABEL_4:
      v2 = 0;
      goto LABEL_5;
    }
    v30 = (__int64 *)(a1 + 72);
    if ( (unsigned __int8)sub_A74710((_QWORD *)(a1 + 72), 0, 43)
      || (v14 = *(_QWORD *)(a1 - 32)) != 0
      && !*(_BYTE *)v14
      && *(_QWORD *)(v14 + 24) == *(_QWORD *)(a1 + 80)
      && (v32 = *(_QWORD *)(v14 + 120), (unsigned __int8)sub_A74710(&v32, 0, 43)) )
    {
      if ( (unsigned __int8)sub_A74710(v30, 0, 90) )
      {
        v2 = 0;
      }
      else
      {
        v21 = *(_QWORD *)(a1 - 32);
        if ( v21 && !*(_BYTE *)v21 && *(_QWORD *)(v21 + 24) == *(_QWORD *)(a1 + 80) )
        {
          v32 = *(_QWORD *)(v21 + 120);
          v2 = sub_A74710(&v32, 0, 90) ^ 1;
        }
      }
      v28 = v34;
      if ( v34 <= 0x40 )
      {
        v22 = (__int64)v33;
      }
      else
      {
        v22 = -1;
        if ( v28 - (unsigned int)sub_C444A0((__int64)&v33) <= 0x40 )
          v22 = *v33;
      }
      v23 = (__int64 *)sub_BD5C60(a1);
      v18 = sub_A77A80(v23, v22);
    }
    else
    {
      if ( (unsigned __int8)sub_A74710(v30, 0, 91) )
      {
        v2 = 0;
      }
      else
      {
        v15 = *(_QWORD *)(a1 - 32);
        if ( v15 && !*(_BYTE *)v15 && *(_QWORD *)(v15 + 24) == *(_QWORD *)(a1 + 80) )
        {
          v32 = *(_QWORD *)(v15 + 120);
          v2 = sub_A74710(&v32, 0, 91) ^ 1;
        }
      }
      v27 = v34;
      if ( v34 <= 0x40 )
      {
        v16 = (__int64)v33;
      }
      else
      {
        v16 = -1;
        if ( v27 - (unsigned int)sub_C444A0((__int64)&v33) <= 0x40 )
          v16 = *v33;
      }
      v17 = (__int64 *)sub_BD5C60(a1);
      v18 = sub_A77A90(v17, v16);
    }
    v19 = v18;
    v20 = (__int64 *)sub_BD5C60(a1);
    *(_QWORD *)(a1 + 72) = sub_A7B440(v30, v20, 0, v19);
  }
LABEL_5:
  v3 = sub_D5CD40(a1, a2);
  v4 = v3;
  if ( v3 && *(_BYTE *)v3 == 17 )
  {
    v5 = *(_DWORD *)(v3 + 32);
    if ( v5 > 0x40 )
    {
      if ( v5 - (unsigned int)sub_C444A0(v3 + 24) > 0x40 )
        goto LABEL_19;
      v6 = **(_QWORD **)(v4 + 24);
      if ( v6 > 0xFFFFFFFF )
        goto LABEL_19;
    }
    else
    {
      v6 = *(_QWORD *)(v3 + 24);
      if ( v6 > 0xFFFFFFFF )
        goto LABEL_19;
    }
    if ( v6 && (v6 & (v6 - 1)) == 0 )
    {
      v7 = sub_A74820((_QWORD *)(a1 + 72));
      v8 = (__int64 *)(a1 + 72);
      v9 = v7;
      if ( !HIBYTE(v7) )
      {
        v10 = *(_QWORD *)(a1 - 32);
        if ( !v10 || (v9 = *(_BYTE *)v10) != 0 )
        {
          v9 = 0;
        }
        else if ( *(_QWORD *)(v10 + 24) == *(_QWORD *)(a1 + 80) )
        {
          v32 = *(_QWORD *)(v10 + 120);
          v11 = sub_A74820(&v32);
          v8 = (__int64 *)(a1 + 72);
          if ( HIBYTE(v11) )
            v9 = v11;
        }
      }
      _BitScanReverse64(&v6, v6);
      v12 = 63 - (v6 ^ 0x3F);
      if ( v12 > v9 )
      {
        v31 = v8;
        v2 = 1;
        v24 = (__int64 *)sub_BD5C60(a1);
        v25 = sub_A77A40(v24, v12);
        v26 = (__int64 *)sub_BD5C60(a1);
        *(_QWORD *)(a1 + 72) = sub_A7B440(v31, v26, 0, v25);
      }
    }
  }
LABEL_19:
  if ( v35 )
  {
    v35 = 0;
    if ( v34 > 0x40 )
    {
      if ( v33 )
        j_j___libc_free_0_0(v33);
    }
  }
  return v2;
}
