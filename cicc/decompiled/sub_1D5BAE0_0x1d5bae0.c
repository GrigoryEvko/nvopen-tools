// Function: sub_1D5BAE0
// Address: 0x1d5bae0
//
__int64 __fastcall sub_1D5BAE0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 **a4)
{
  unsigned __int8 *v6; // rbx
  _QWORD *v7; // rax
  _QWORD *v8; // r13
  __int64 v9; // rax
  unsigned __int8 *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // eax
  _QWORD *v14; // rdx
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  unsigned __int8 *v22; // [rsp+8h] [rbp-C8h] BYREF
  char *v23; // [rsp+10h] [rbp-C0h] BYREF
  char v24; // [rsp+20h] [rbp-B0h]
  char v25; // [rsp+21h] [rbp-AFh]
  unsigned __int8 *v26[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v27; // [rsp+40h] [rbp-90h]
  unsigned __int8 *v28; // [rsp+50h] [rbp-80h] BYREF
  __int64 v29; // [rsp+58h] [rbp-78h]
  __int64 *v30; // [rsp+60h] [rbp-70h]
  __int64 v31; // [rsp+68h] [rbp-68h]
  __int64 v32; // [rsp+70h] [rbp-60h]
  int v33; // [rsp+78h] [rbp-58h]
  __int64 v34; // [rsp+80h] [rbp-50h]
  __int64 v35; // [rsp+88h] [rbp-48h]

  v6 = a2;
  v7 = (_QWORD *)sub_22077B0(24);
  v8 = v7;
  if ( !v7 )
  {
    a3 = MEMORY[0x10];
    goto LABEL_12;
  }
  v7[1] = a2;
  *v7 = off_4985618;
  v9 = sub_16498A0((__int64)a2);
  v10 = (unsigned __int8 *)*((_QWORD *)a2 + 6);
  v28 = 0;
  v31 = v9;
  v11 = *((_QWORD *)v6 + 5);
  v32 = 0;
  v29 = v11;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v30 = (__int64 *)(v6 + 24);
  v26[0] = v10;
  if ( v10 )
  {
    sub_1623A60((__int64)v26, (__int64)v10, 2);
    if ( v28 )
      sub_161E7C0((__int64)&v28, (__int64)v28);
    v28 = v26[0];
    if ( v26[0] )
      sub_1623210((__int64)v26, v26[0], (__int64)&v28);
  }
  v25 = 1;
  v23 = "promoted";
  v24 = 3;
  if ( a4 != *(__int64 ***)a3 )
  {
    if ( *(_BYTE *)(a3 + 16) <= 0x10u )
    {
      v12 = sub_15A46C0(37, (__int64 ***)a3, a4, 0);
      a2 = v28;
      a3 = v12;
      goto LABEL_10;
    }
    v27 = 257;
    v16 = sub_15FDBD0(37, a3, (__int64)a4, (__int64)v26, 0);
    a3 = v16;
    if ( v29 )
    {
      v17 = v30;
      sub_157E9D0(v29 + 40, v16);
      v18 = *(_QWORD *)(a3 + 24);
      v19 = *v17;
      *(_QWORD *)(a3 + 32) = v17;
      v19 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a3 + 24) = v19 | v18 & 7;
      *(_QWORD *)(v19 + 8) = a3 + 24;
      *v17 = *v17 & 7 | (a3 + 24);
    }
    sub_164B780(a3, (__int64 *)&v23);
    a2 = v28;
    if ( !v28 )
    {
      v8[2] = a3;
      v13 = *(_DWORD *)(a1 + 8);
      if ( v13 < *(_DWORD *)(a1 + 12) )
        goto LABEL_13;
      goto LABEL_26;
    }
    v22 = v28;
    sub_1623A60((__int64)&v22, (__int64)v28, 2);
    v20 = *(_QWORD *)(a3 + 48);
    if ( v20 )
      sub_161E7C0(a3 + 48, v20);
    v21 = v22;
    *(_QWORD *)(a3 + 48) = v22;
    if ( v21 )
      sub_1623210((__int64)&v22, v21, a3 + 48);
  }
  a2 = v28;
LABEL_10:
  v8[2] = a3;
  if ( a2 )
  {
    sub_161E7C0((__int64)&v28, (__int64)a2);
    a3 = v8[2];
  }
LABEL_12:
  v13 = *(_DWORD *)(a1 + 8);
  if ( v13 < *(_DWORD *)(a1 + 12) )
    goto LABEL_13;
LABEL_26:
  sub_1D5B850(a1, (__int64)a2);
  v13 = *(_DWORD *)(a1 + 8);
LABEL_13:
  v14 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v13);
  if ( v14 )
  {
    *v14 = v8;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    *(_DWORD *)(a1 + 8) = v13 + 1;
    (*(void (__fastcall **)(_QWORD *))(*v8 + 8LL))(v8);
  }
  return a3;
}
