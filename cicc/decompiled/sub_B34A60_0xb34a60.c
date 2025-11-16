// Function: sub_B34A60
// Address: 0xb34a60
//
__int64 __fastcall sub_B34A60(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  int v4; // eax
  __int64 v5; // rdx
  __int16 v6; // ax
  unsigned int v8; // r12d
  __int64 *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 *v13; // rsi
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // [rsp-8h] [rbp-88h]
  __int64 v21; // [rsp+8h] [rbp-78h] BYREF
  unsigned __int8 *v22; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v23; // [rsp+18h] [rbp-68h]
  int v24; // [rsp+1Ch] [rbp-64h]
  _DWORD v25[8]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v26; // [rsp+40h] [rbp-40h]

  v26 = 257;
  v2 = *((_QWORD *)a2 + 1);
  v24 = 0;
  v22 = a2;
  v21 = v2;
  v3 = sub_B33D10(a1, 0x161u, (__int64)&v21, 1, (int)&v22, 1, v23, (__int64)v25);
  v4 = *a2;
  v5 = (unsigned int)(v4 - 2);
  if ( (unsigned __int8)(v4 - 2) <= 1u || !(_BYTE)v4 )
  {
    v6 = (*((_WORD *)a2 + 17) >> 1) & 0x3F;
    if ( !v6 )
      return v3;
LABEL_7:
    v8 = (unsigned __int8)(v6 - 1);
    v9 = (__int64 *)sub_BD5C60(v3, v20, v5);
    v10 = sub_A77A40(v9, v8);
    v25[0] = 0;
    v11 = v10;
    v13 = (__int64 *)sub_BD5C60(v3, v8, v12);
    *(_QWORD *)(v3 + 72) = sub_A7B660((__int64 *)(v3 + 72), v13, v25, 1, v11);
    v15 = (__int64 *)sub_BD5C60(v3, v13, v14);
    v16 = v8;
    v17 = sub_A77A40(v15, v8);
    v19 = (__int64 *)sub_BD5C60(v3, v16, v18);
    *(_QWORD *)(v3 + 72) = sub_A7B440((__int64 *)(v3 + 72), v19, 0, v17);
    return v3;
  }
  if ( (_BYTE)v4 == 1 )
  {
    v6 = (*(_WORD *)(sub_B325F0((__int64)a2) + 34) >> 1) & 0x3F;
    if ( v6 )
      goto LABEL_7;
  }
  return v3;
}
