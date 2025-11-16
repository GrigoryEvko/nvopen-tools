// Function: sub_29E04C0
// Address: 0x29e04c0
//
_QWORD *__fastcall sub_29E04C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int16 a6, char a7)
{
  unsigned __int8 v7; // r13
  _QWORD *v8; // r12
  __int64 v10; // r14
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r10
  __int64 v15; // r8
  __int64 v16; // rax
  unsigned int v17; // eax
  unsigned __int8 v18; // r8
  __int64 v19; // rax
  unsigned int v20; // r13d
  unsigned int v21; // r13d
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  char v29; // [rsp+Bh] [rbp-85h]
  char v32; // [rsp+20h] [rbp-70h]
  __int64 v33; // [rsp+28h] [rbp-68h]
  unsigned __int8 v34; // [rsp+28h] [rbp-68h]
  _QWORD v35[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v36; // [rsp+50h] [rbp-40h]

  v7 = a6;
  v8 = (_QWORD *)a2;
  v29 = a6;
  v32 = HIBYTE(a6);
  v10 = sub_B43CB0(a3);
  v33 = sub_B2BEC0(v10);
  v11 = sub_B2DCE0(a4);
  v14 = a3;
  if ( !v11 && !a7 )
  {
    v18 = sub_AE5260(v33, (__int64)a1);
    if ( !v32 )
    {
LABEL_11:
      v19 = *(_QWORD *)(a2 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
        v19 = **(_QWORD **)(v19 + 16);
      v34 = v18;
      v20 = *(_DWORD *)(v19 + 8);
      v35[0] = sub_BD5D20(a2);
      v21 = v20 >> 8;
      v36 = 261;
      v35[1] = v22;
      v23 = sub_BD2C40(80, 1u);
      v8 = v23;
      if ( v23 )
        sub_B4CCA0((__int64)v23, a1, v21, 0, v34, (__int64)v35, 0, 0);
      v24 = *(_QWORD *)(v10 + 80);
      if ( !v24 )
        BUG();
      sub_B44220(v8, *(_QWORD *)(v24 + 32), 1);
      v27 = *(unsigned int *)(a5 + 48);
      if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 52) )
      {
        sub_C8D5F0(a5 + 40, (const void *)(a5 + 56), v27 + 1, 8u, v25, v26);
        v27 = *(unsigned int *)(a5 + 48);
      }
      *(_QWORD *)(*(_QWORD *)(a5 + 40) + 8 * v27) = v8;
      ++*(_DWORD *)(a5 + 48);
      return v8;
    }
LABEL_9:
    if ( v18 < v7 )
      v18 = v7;
    goto LABEL_11;
  }
  if ( v32 && v7 )
  {
    v15 = 0;
    if ( *(_QWORD *)a5 )
    {
      v16 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))a5)(
              *(_QWORD *)(a5 + 8),
              v10,
              v12,
              v13,
              0);
      v14 = a3;
      v15 = v16;
    }
    v17 = 256;
    LOBYTE(v17) = v29;
    if ( (unsigned __int8)sub_F518D0((unsigned __int8 *)a2, v17, v33, v14, v15, 0) < v7 )
    {
      v18 = sub_AE5260(v33, (__int64)a1);
      goto LABEL_9;
    }
  }
  return v8;
}
