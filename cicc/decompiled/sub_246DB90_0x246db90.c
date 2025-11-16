// Function: sub_246DB90
// Address: 0x246db90
//
__int64 __fastcall sub_246DB90(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v8; // ebx
  __int64 v9; // rax
  unsigned int v10; // eax
  char v11; // dl
  unsigned int v12; // eax
  unsigned int v13; // edx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  _BYTE *v20; // rax
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // rax
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp-10h] [rbp-B0h]
  __int64 v31; // [rsp+8h] [rbp-98h]
  unsigned __int64 v32; // [rsp+10h] [rbp-90h]
  unsigned __int64 v33; // [rsp+18h] [rbp-88h]
  _QWORD v34[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v35[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v36; // [rsp+60h] [rbp-40h]

  v8 = 4;
  v9 = sub_B2BEC0(*a1);
  v10 = sub_9208B0(v9, *(_QWORD *)(a3 + 8));
  if ( !v11 )
  {
    v8 = 0;
    if ( v10 > 8 )
    {
      v12 = ((v10 + 7) >> 3) - 1;
      v8 = v12;
      if ( v12 )
      {
        _BitScanReverse(&v13, v12);
        v8 = 32 - (v13 ^ 0x1F);
      }
    }
  }
  if ( *(_BYTE *)a3 <= 0x15u
    || (v14 = (int)qword_4FE8148, v15 = a1[209] + 1, a1[209] = v15, (int)v14 < 0)
    || v8 > 3
    || v15 <= v14
    || (v20 = (_BYTE *)a1[1], *v20) )
  {
    v35[0] = "_mscmp";
    v36 = 259;
    v16 = sub_2465600((__int64)a1, a3, a2, (__int64)v35);
    v17 = *(_QWORD *)(a2 + 56);
    if ( v17 )
      v17 -= 24;
    v18 = sub_F38250(v16, (__int64 *)(v17 + 24), 0, *(_BYTE *)(a1[1] + 8) ^ 1u, *(_QWORD *)(a1[1] + 728), 0, 0, 0);
    sub_D5F1F0(a2, v18);
    return sub_246D400((__int64)a1, a2, a4);
  }
  else
  {
    v21 = (unsigned __int64 *)&v20[16 * v8 + 184];
    v32 = *v21;
    v33 = v21[1];
    v22 = sub_24650D0((__int64)a1, a3, a2);
    v23 = *(_QWORD **)(a2 + 72);
    v36 = 257;
    v31 = v22;
    v24 = sub_BCD140(v23, 8 << v8);
    v34[0] = sub_A82F30((unsigned int **)a2, v31, v24, (__int64)v35, 0);
    v25 = a1[1];
    v36 = 257;
    if ( !*(_DWORD *)(v25 + 4) || !a4 )
    {
      v29 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
      a4 = sub_ACD640(v29, 0, 0);
    }
    v34[1] = a4;
    v26 = sub_921880((unsigned int **)a2, v32, v33, (int)v34, 2, (__int64)v35, 0);
    v27 = (__int64 *)sub_BD5C60(v26);
    *(_QWORD *)(v26 + 72) = sub_A7A090((__int64 *)(v26 + 72), v27, 1, 79);
    v28 = (__int64 *)sub_BD5C60(v26);
    *(_QWORD *)(v26 + 72) = sub_A7A090((__int64 *)(v26 + 72), v28, 2, 79);
    return v30;
  }
}
