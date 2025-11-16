// Function: sub_20351B0
// Address: 0x20351b0
//
__int64 __fastcall sub_20351B0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // r13
  char *v9; // rdx
  char v10; // al
  const void **v11; // rdx
  __int64 v12; // rcx
  const void **v13; // r8
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rsi
  __int64 *v20; // r10
  const void **v21; // r8
  unsigned int v22; // ebx
  __int64 v23; // r12
  __int64 v25; // rax
  const void **v26; // rdx
  __int128 v27; // [rsp-10h] [rbp-80h]
  __int128 v28; // [rsp-10h] [rbp-80h]
  const void **v29; // [rsp+0h] [rbp-70h]
  __int64 v30; // [rsp+8h] [rbp-68h]
  const void **v31; // [rsp+10h] [rbp-60h]
  __int64 *v32; // [rsp+18h] [rbp-58h]
  __int64 *v33; // [rsp+18h] [rbp-58h]
  __int64 v34; // [rsp+20h] [rbp-50h] BYREF
  int v35; // [rsp+28h] [rbp-48h]
  __int64 v36; // [rsp+30h] [rbp-40h] BYREF
  const void **v37; // [rsp+38h] [rbp-38h]

  v6 = sub_2032580(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = v7;
  v9 = *(char **)(a2 + 40);
  v32 = *(__int64 **)(a1 + 8);
  v10 = *v9;
  v11 = (const void **)*((_QWORD *)v9 + 1);
  LOBYTE(v36) = v10;
  v37 = v11;
  if ( v10 )
  {
    if ( (unsigned __int8)(v10 - 14) > 0x5Fu )
    {
LABEL_3:
      v12 = v36;
      v13 = v37;
      goto LABEL_4;
    }
  }
  else if ( !sub_1F58D20((__int64)&v36) )
  {
    goto LABEL_3;
  }
  LOBYTE(v25) = sub_1F7E0F0((__int64)&v36);
  v12 = v25;
  v13 = v26;
LABEL_4:
  v14 = *(_QWORD *)(a2 + 72);
  v34 = v14;
  if ( v14 )
  {
    v29 = v13;
    v30 = v12;
    sub_1623A60((__int64)&v34, v14, 2);
    v13 = v29;
    v12 = v30;
  }
  *((_QWORD *)&v27 + 1) = v8;
  v15 = *(unsigned __int16 *)(a2 + 24);
  *(_QWORD *)&v27 = v6;
  v35 = *(_DWORD *)(a2 + 64);
  v16 = sub_1D309E0(v32, v15, (__int64)&v34, v12, v13, 0, a3, a4, a5, v27);
  v18 = v17;
  if ( v34 )
    sub_161E7C0((__int64)&v34, v34);
  v19 = *(_QWORD *)(a2 + 72);
  v20 = *(__int64 **)(a1 + 8);
  v21 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v22 = **(unsigned __int8 **)(a2 + 40);
  v36 = v19;
  if ( v19 )
  {
    v31 = v21;
    v33 = v20;
    sub_1623A60((__int64)&v36, v19, 2);
    v21 = v31;
    v20 = v33;
  }
  *((_QWORD *)&v28 + 1) = v18;
  *(_QWORD *)&v28 = v16;
  LODWORD(v37) = *(_DWORD *)(a2 + 64);
  v23 = sub_1D309E0(v20, 111, (__int64)&v36, v22, v21, 0, a3, a4, a5, v28);
  if ( v36 )
    sub_161E7C0((__int64)&v36, v36);
  return v23;
}
