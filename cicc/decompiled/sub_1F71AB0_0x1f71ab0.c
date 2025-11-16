// Function: sub_1F71AB0
// Address: 0x1f71ab0
//
__int64 __fastcall sub_1F71AB0(__int64 *a1, _QWORD *a2, __m128i a3, double a4, __m128i a5)
{
  _QWORD *v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // r12
  unsigned __int64 v10; // r15
  char *v11; // rax
  __int64 v12; // r13
  __int64 *v13; // rsi
  char v14; // r8
  const void **v15; // rax
  __int64 *v17; // r12
  __int64 v18; // rbx
  __int64 v19; // r13
  unsigned __int8 *v20; // rax
  __int64 v21; // rax
  const void **v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rax
  char v25; // di
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rcx
  const void **v29; // r8
  __int128 v30; // rax
  unsigned int v31; // [rsp+Ch] [rbp-84h]
  _QWORD *v32; // [rsp+10h] [rbp-80h]
  __int64 v33; // [rsp+10h] [rbp-80h]
  __int64 v34; // [rsp+18h] [rbp-78h]
  __int64 v35; // [rsp+18h] [rbp-78h]
  char v36; // [rsp+20h] [rbp-70h]
  char v37; // [rsp+20h] [rbp-70h]
  const void **v38; // [rsp+20h] [rbp-70h]
  __int64 v39; // [rsp+28h] [rbp-68h]
  unsigned int v40; // [rsp+30h] [rbp-60h] BYREF
  const void **v41; // [rsp+38h] [rbp-58h]
  __int64 *v42; // [rsp+40h] [rbp-50h] BYREF
  int v43; // [rsp+48h] [rbp-48h]
  char v44[8]; // [rsp+50h] [rbp-40h] BYREF
  __int64 v45; // [rsp+58h] [rbp-38h]

  v5 = a2;
  v6 = a2[4];
  v7 = *(_QWORD *)v6;
  v8 = *(_QWORD *)v6;
  v9 = *(_QWORD *)(v6 + 40);
  v31 = *(_DWORD *)(v6 + 8);
  v10 = *(_QWORD *)(v6 + 8);
  v11 = (char *)a2[5];
  v39 = v7;
  v12 = v9;
  v13 = (__int64 *)a2[9];
  v14 = *v11;
  v15 = (const void **)*((_QWORD *)v11 + 1);
  v42 = v13;
  LOBYTE(v40) = v14;
  v41 = v15;
  if ( v13 )
  {
    v32 = v5;
    v36 = v14;
    sub_1623A60((__int64)&v42, (__int64)v13, 2);
    v5 = v32;
    v14 = v36;
  }
  v43 = *((_DWORD *)v5 + 16);
  if ( v14 )
  {
    if ( (unsigned __int8)(v14 - 14) > 0x5Fu )
      goto LABEL_10;
  }
  else if ( !sub_1F58D20((__int64)&v40) )
  {
    goto LABEL_10;
  }
  if ( (unsigned __int8)sub_1D16620(v9, v13) )
    goto LABEL_6;
  if ( (unsigned __int8)sub_1D16620(v39, v13) )
  {
    v9 = v8;
    goto LABEL_6;
  }
LABEL_10:
  if ( !sub_1D185B0(v9) )
  {
    if ( sub_1D18910(v9) )
    {
      v17 = (__int64 *)*a1;
      v18 = 16LL * v31;
      v19 = a1[1];
      v20 = (unsigned __int8 *)(v18 + *(_QWORD *)(v39 + 40));
      v37 = *((_BYTE *)a1 + 25);
      v33 = *((_QWORD *)v20 + 1);
      v34 = *v20;
      v21 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32));
      v23 = sub_1F40B60(v19, v34, v33, v21, v37);
      v24 = v18 + *(_QWORD *)(v39 + 40);
      v25 = *(_BYTE *)v24;
      v26 = *(_QWORD *)(v24 + 8);
      v44[0] = v25;
      v45 = v26;
      if ( v25 )
      {
        v27 = sub_1F6C8D0(v25);
      }
      else
      {
        v35 = v23;
        v38 = v22;
        v27 = sub_1F58D40((__int64)v44);
        v28 = v35;
        v29 = v38;
      }
      *(_QWORD *)&v30 = sub_1D38BB0((__int64)v17, (unsigned int)(v27 - 1), (__int64)&v42, v28, v29, 0, a3, a4, a5, 0);
      v9 = (__int64)sub_1D332F0(
                      v17,
                      123,
                      (__int64)&v42,
                      *(unsigned __int8 *)(*(_QWORD *)(v39 + 40) + 16LL * v31),
                      *(const void ***)(*(_QWORD *)(v39 + 40) + 16LL * v31 + 8),
                      0,
                      *(double *)a3.m128i_i64,
                      a4,
                      a5,
                      v8,
                      v10,
                      v30);
    }
    else if ( *(_WORD *)(v39 + 24) == 48 || (v9 = 0, *(_WORD *)(v12 + 24) == 48) )
    {
      v9 = sub_1D38BB0(*a1, 0, (__int64)&v42, v40, v41, 0, a3, a4, a5, 0);
    }
  }
LABEL_6:
  if ( v42 )
    sub_161E7C0((__int64)&v42, (__int64)v42);
  return v9;
}
