// Function: sub_293B690
// Address: 0x293b690
//
__int64 __fastcall sub_293B690(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v8; // r14
  unsigned __int64 v9; // r13
  char v10; // dl
  __int64 v11; // rdx
  unsigned __int64 v12; // r13
  char v13; // dl
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-90h]
  __int64 v19; // [rsp+8h] [rbp-88h]
  char v20; // [rsp+17h] [rbp-79h]
  char v21; // [rsp+17h] [rbp-79h]
  __int64 v22; // [rsp+18h] [rbp-78h]
  unsigned __int64 v23; // [rsp+20h] [rbp-70h] BYREF
  __int64 v24; // [rsp+28h] [rbp-68h]
  _QWORD v25[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v26; // [rsp+40h] [rbp-50h]
  __int64 v27; // [rsp+48h] [rbp-48h]
  char v28; // [rsp+50h] [rbp-40h]

  sub_2939E80((__int64)v25, a2, a3);
  if ( !v28 )
    goto LABEL_2;
  v8 = v26;
  v22 = v27;
  v18 = v25[1];
  v19 = v25[0];
  v9 = (sub_9208B0(a5, v26) + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v20 = v10;
  v23 = sub_9208B0(a5, v8);
  v24 = v11;
  if ( v23 != v9 )
    goto LABEL_2;
  if ( (_BYTE)v24 == v20
    && (!v22
     || (v12 = (sub_9208B0(a5, v22) + 7) & 0xFFFFFFFFFFFFFFF8LL,
         v21 = v13,
         v23 = sub_9208B0(a5, v22),
         v24 = v14,
         v23 == v12)
     && (_BYTE)v24 == v21) )
  {
    v15 = sub_9208B0(a5, v8);
    v24 = v16;
    v23 = (unsigned __int64)(v15 + 7) >> 3;
    v17 = sub_CA1930(&v23);
    *(_QWORD *)(a1 + 16) = v8;
    *(_BYTE *)(a1 + 32) = a4;
    *(_QWORD *)a1 = v19;
    *(_QWORD *)(a1 + 40) = v17;
    *(_QWORD *)(a1 + 8) = v18;
    *(_BYTE *)(a1 + 48) = 1;
    *(_QWORD *)(a1 + 24) = v22;
  }
  else
  {
LABEL_2:
    *(_QWORD *)(a1 + 48) = 0;
  }
  return a1;
}
