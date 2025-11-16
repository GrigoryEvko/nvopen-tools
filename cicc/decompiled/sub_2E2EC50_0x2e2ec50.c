// Function: sub_2E2EC50
// Address: 0x2e2ec50
//
bool __fastcall sub_2E2EC50(__int64 **a1, __int64 a2)
{
  __int64 *v2; // r14
  _BYTE *v3; // rdx
  _BYTE *v4; // rbx
  bool result; // al
  __int64 v6; // rdx
  __int64 v7; // r9
  __int16 v8; // ax
  unsigned __int8 v9; // al
  _BYTE *v10; // rdx
  _BYTE *v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int8 v16; // r12
  unsigned __int8 v17; // al
  __int64 v18; // [rsp+0h] [rbp-110h]
  __int64 v19; // [rsp+0h] [rbp-110h]
  __int64 v20; // [rsp+8h] [rbp-108h]
  __int64 v21; // [rsp+8h] [rbp-108h]
  __int64 v22; // [rsp+10h] [rbp-100h]
  unsigned __int8 v23; // [rsp+1Fh] [rbp-F1h]
  __int64 **v24; // [rsp+20h] [rbp-F0h]
  __int64 v25; // [rsp+28h] [rbp-E8h]
  int v26; // [rsp+30h] [rbp-E0h]
  _BYTE *v27; // [rsp+30h] [rbp-E0h]
  _BYTE *v28; // [rsp+38h] [rbp-D8h]
  __int64 v29; // [rsp+38h] [rbp-D8h]
  __int64 v30; // [rsp+40h] [rbp-D0h]
  __int64 **v31; // [rsp+48h] [rbp-C8h]
  bool v32; // [rsp+48h] [rbp-C8h]
  __int64 *v33; // [rsp+50h] [rbp-C0h] BYREF
  unsigned __int64 v34; // [rsp+58h] [rbp-B8h]
  __int64 v35; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v36[4]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v37; // [rsp+90h] [rbp-80h] BYREF
  unsigned __int64 v38; // [rsp+98h] [rbp-78h]
  __int64 v39; // [rsp+A0h] [rbp-70h] BYREF
  const char *v40; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v41; // [rsp+B8h] [rbp-58h]
  const char *v42; // [rsp+C0h] [rbp-50h]
  _BYTE *v43; // [rsp+C8h] [rbp-48h]
  __int16 v44; // [rsp+D0h] [rbp-40h]

  v2 = *a1;
  v31 = (__int64 **)sub_BCE3C0(*a1, 0);
  v44 = 1283;
  v40 = "__emutls_v.";
  v42 = sub_BD5D20(a2);
  v43 = v3;
  sub_CA0F50((__int64 *)&v33, (void **)&v40);
  v4 = sub_BA8CD0((__int64)a1, (__int64)v33, v34, 1);
  result = 0;
  if ( !v4 )
  {
    v30 = (__int64)(a1 + 39);
    v25 = sub_AC9EC0(v31);
    if ( !sub_B2FC80(a2) )
    {
      v6 = *(_QWORD *)(a2 - 32);
      if ( *(_BYTE *)v6 == 17 )
      {
        if ( *(_DWORD *)(v6 + 32) <= 0x40u )
        {
          if ( *(_QWORD *)(v6 + 24) )
            v4 = *(_BYTE **)(a2 - 32);
        }
        else
        {
          v26 = *(_DWORD *)(v6 + 32);
          v28 = *(_BYTE **)(a2 - 32);
          if ( v26 != (unsigned int)sub_C444A0(v6 + 24) )
            v4 = v28;
        }
      }
      else if ( *(_BYTE *)v6 != 14 )
      {
        v4 = *(_BYTE **)(a2 - 32);
      }
    }
    v29 = sub_AE4420(v30, (__int64)v2, 0);
    v36[3] = sub_BCE3C0(v2, 0);
    v36[0] = v29;
    v36[1] = v29;
    v36[2] = (__int64)v31;
    v24 = (__int64 **)sub_BD0EF0(v36, 4);
    v27 = sub_BA8D60((__int64)a1, (__int64)v33, v34, (__int64)v24);
    sub_2E2EB00((__int64)a1, a2, (__int64)v27);
    result = sub_B2FC80(a2);
    if ( !result )
    {
      v7 = *(_QWORD *)(a2 + 24);
      v8 = (*(_WORD *)(a2 + 34) >> 1) & 0x3F;
      if ( v8 )
      {
        v23 = v8 - 1;
      }
      else
      {
        v22 = *(_QWORD *)(a2 + 24);
        v9 = sub_AE5020(v30, v22);
        v7 = v22;
        v23 = v9;
      }
      if ( v4 )
      {
        v20 = v7;
        v42 = sub_BD5D20(a2);
        v44 = 1283;
        v40 = "__emutls_t.";
        v43 = v10;
        sub_CA0F50((__int64 *)&v37, (void **)&v40);
        v11 = sub_BA8D60((__int64)a1, (__int64)v37, v38, v20);
        if ( !v11 || *v11 != 3 )
        {
          MEMORY[0x50] &= ~1u;
          BUG();
        }
        v11[80] |= 1u;
        v18 = (__int64)v11;
        sub_B30160((__int64)v11, (__int64)v4);
        sub_B2F770(v18, v23);
        sub_2E2EB00((__int64)a1, a2, v18);
        v12 = (_BYTE *)v18;
        v7 = v20;
        if ( v37 != &v39 )
        {
          v21 = v18;
          v19 = v7;
          j_j___libc_free_0((unsigned __int64)v37);
          v7 = v19;
          v12 = (_BYTE *)v21;
        }
        v4 = v12;
      }
      v40 = (const char *)sub_9208B0(v30, v7);
      v41 = v13;
      v37 = (__int64 *)((unsigned __int64)(v40 + 7) >> 3);
      LOBYTE(v38) = v13;
      v14 = sub_CA1930(&v37);
      v40 = (const char *)sub_ACD640(v29, v14, 0);
      v41 = sub_ACD640(v29, 1LL << v23, 0);
      if ( !v4 )
        v4 = (_BYTE *)v25;
      v42 = (const char *)v25;
      v43 = v4;
      v15 = sub_AD24A0(v24, (__int64 *)&v40, 4);
      sub_B30160((__int64)v27, v15);
      v16 = sub_AE5020(v30, (__int64)v31);
      v17 = sub_AE5020(v30, v29);
      if ( v17 < v16 )
        v17 = v16;
      sub_B2F770((__int64)v27, v17);
      result = 1;
    }
  }
  if ( v33 != &v35 )
  {
    v32 = result;
    j_j___libc_free_0((unsigned __int64)v33);
    return v32;
  }
  return result;
}
