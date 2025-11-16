// Function: sub_17FE280
// Address: 0x17fe280
//
__int64 __fastcall sub_17FE280(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 **v5; // rcx
  char v6; // al
  char v8; // al
  int v9; // edi
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rdi
  _QWORD *v14; // r12
  unsigned __int64 *v15; // r14
  __int64 v16; // rdi
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  int v22; // edi
  __int64 v23; // rax
  unsigned __int8 *v24; // [rsp+8h] [rbp-48h] BYREF
  _BYTE v25[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v26; // [rsp+20h] [rbp-30h]

  v5 = *(__int64 ***)a2;
  if ( a3 == *(_QWORD *)a2 )
    return a2;
  v6 = *((_BYTE *)v5 + 8);
  if ( v6 == 16 )
    v6 = *(_BYTE *)(*v5[2] + 8);
  if ( v6 == 15 )
  {
    v11 = *(_BYTE *)(a3 + 8);
    if ( v11 == 16 )
      v11 = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
    if ( v11 != 11 )
      goto LABEL_9;
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
      v9 = 45;
      return sub_15A46C0(v9, (__int64 ***)a2, (__int64 **)a3, 0);
    }
    v26 = 257;
    v22 = 45;
  }
  else
  {
    if ( v6 != 11 )
      goto LABEL_9;
    v8 = *(_BYTE *)(a3 + 8);
    if ( v8 == 16 )
      v8 = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
    if ( v8 != 15 )
    {
LABEL_9:
      if ( *(_BYTE *)(a2 + 16) <= 0x10u )
      {
        v9 = 47;
        return sub_15A46C0(v9, (__int64 ***)a2, (__int64 **)a3, 0);
      }
      v26 = 257;
      v12 = sub_15FDBD0(47, a2, a3, (__int64)v25, 0);
      v13 = a1[1];
      v14 = (_QWORD *)v12;
      if ( !v13 )
        goto LABEL_20;
      v15 = (unsigned __int64 *)a1[2];
      v16 = v13 + 40;
      goto LABEL_19;
    }
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
      v9 = 46;
      return sub_15A46C0(v9, (__int64 ***)a2, (__int64 **)a3, 0);
    }
    v22 = 46;
    v26 = 257;
  }
  v14 = (_QWORD *)sub_15FDBD0(v22, a2, a3, (__int64)v25, 0);
  v23 = a1[1];
  if ( !v23 )
    goto LABEL_20;
  v15 = (unsigned __int64 *)a1[2];
  v16 = v23 + 40;
LABEL_19:
  sub_157E9D0(v16, (__int64)v14);
  v17 = *v15;
  v18 = v14[3];
  v14[4] = v15;
  v17 &= 0xFFFFFFFFFFFFFFF8LL;
  v14[3] = v17 | v18 & 7;
  *(_QWORD *)(v17 + 8) = v14 + 3;
  *v15 = *v15 & 7 | (unsigned __int64)(v14 + 3);
LABEL_20:
  sub_164B780((__int64)v14, a4);
  v19 = *a1;
  if ( *a1 )
  {
    v24 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v24, v19, 2);
    v20 = v14[6];
    if ( v20 )
      sub_161E7C0((__int64)(v14 + 6), v20);
    v21 = v24;
    v14[6] = v24;
    if ( v21 )
      sub_1623210((__int64)&v24, v21, (__int64)(v14 + 6));
  }
  return (__int64)v14;
}
