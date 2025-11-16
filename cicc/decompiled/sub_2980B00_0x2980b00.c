// Function: sub_2980B00
// Address: 0x2980b00
//
void __fastcall sub_2980B00(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 *v7; // rax
  __int64 v8; // r9
  _BYTE *v9; // r8
  __int64 v10; // rcx
  __int64 v11; // rdx
  _BYTE *v12; // r8
  __int64 v13; // r14
  unsigned int v14; // eax
  __int64 v15; // r9
  __int64 *v16; // rax
  __int64 v17; // rax
  _BYTE *v18; // r8
  __int64 v19; // r14
  __int64 *v20; // rax
  _BYTE *v21; // r14
  __int64 *v22; // rax
  _BYTE *v23; // [rsp+0h] [rbp-60h]
  _BYTE *v24; // [rsp+8h] [rbp-58h]
  _BYTE *v25; // [rsp+8h] [rbp-58h]
  _BYTE *v26; // [rsp+8h] [rbp-58h]
  unsigned __int64 v27; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+18h] [rbp-48h]
  unsigned __int64 v29; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+28h] [rbp-38h]

  if ( *a3 == 46 )
  {
    v21 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( v21 )
    {
      v26 = (_BYTE *)*((_QWORD *)a3 - 4);
      if ( *v26 == 17 )
      {
        v22 = sub_DD8400(*(_QWORD *)(a1 + 16), a2);
        v10 = (__int64)v26;
        v8 = a4;
        v9 = v21;
        v11 = (__int64)v22;
        goto LABEL_4;
      }
    }
LABEL_3:
    v6 = sub_ACD640(*(_QWORD *)(a4 + 8), 1, 0);
    v7 = sub_DD8400(*(_QWORD *)(a1 + 16), a2);
    v8 = a4;
    v9 = a3;
    v10 = v6;
    v11 = (__int64)v7;
LABEL_4:
    sub_297F050(a1, 1, v11, v10, v9, v8);
    return;
  }
  if ( *a3 != 54 )
    goto LABEL_3;
  v12 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( !v12 )
    goto LABEL_3;
  v13 = *((_QWORD *)a3 - 4);
  if ( *(_BYTE *)v13 != 17 )
    goto LABEL_3;
  v14 = *(_DWORD *)(v13 + 32);
  v28 = v14;
  if ( v14 <= 0x40 )
  {
    v27 = 1;
    v15 = v13 + 24;
    v30 = v14;
LABEL_9:
    v29 = v27;
    goto LABEL_10;
  }
  v23 = v12;
  sub_C43690((__int64)&v27, 1, 0);
  v15 = v13 + 24;
  v12 = v23;
  v30 = v28;
  if ( v28 <= 0x40 )
    goto LABEL_9;
  sub_C43780((__int64)&v29, (const void **)&v27);
  v12 = v23;
  v15 = v13 + 24;
LABEL_10:
  v24 = v12;
  sub_C47AC0((__int64)&v29, v15);
  v16 = (__int64 *)sub_BD5C60(v13);
  v17 = sub_ACCFD0(v16, (__int64)&v29);
  v18 = v24;
  v19 = v17;
  if ( v30 > 0x40 && v29 )
  {
    j_j___libc_free_0_0(v29);
    v18 = v24;
  }
  v25 = v18;
  v20 = sub_DD8400(*(_QWORD *)(a1 + 16), a2);
  sub_297F050(a1, 1, (__int64)v20, v19, v25, a4);
  if ( v28 > 0x40 )
  {
    if ( v27 )
      j_j___libc_free_0_0(v27);
  }
}
