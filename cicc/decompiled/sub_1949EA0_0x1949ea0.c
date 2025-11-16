// Function: sub_1949EA0
// Address: 0x1949ea0
//
__int64 __fastcall sub_1949EA0(__int64 a1, __int64 a2, char a3, __m128i a4, __m128i a5)
{
  char v6; // r12
  __int64 v8; // r14
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r14
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdx
  _QWORD *v17; // r15
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // r11
  __int64 v22; // rdi
  _QWORD *v23; // r15
  __int64 v24; // rax
  _QWORD *v25; // rbx
  __int64 v26; // rax
  _QWORD *v27; // r12
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  char v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+20h] [rbp-60h]
  __int64 v33; // [rsp+20h] [rbp-60h]
  __int64 v34; // [rsp+20h] [rbp-60h]
  __int64 v35; // [rsp+28h] [rbp-58h]
  __int64 *v36; // [rsp+30h] [rbp-50h] BYREF
  __int64 v37; // [rsp+38h] [rbp-48h]
  __int64 v38; // [rsp+40h] [rbp-40h] BYREF
  __int64 v39; // [rsp+48h] [rbp-38h]

  v6 = a3;
  v8 = sub_1456040(*(_QWORD *)(a2 + 64));
  if ( v8 != sub_1456040(*(_QWORD *)(a2 + 88)) )
  {
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  v32 = sub_146F1B0(*(_QWORD *)(a2 + 16), *(_QWORD *)(a2 + 160));
  v35 = sub_146F1B0(*(_QWORD *)(a2 + 16), *(_QWORD *)(a2 + 176));
  v31 = *(_BYTE *)(a2 + 184);
  v10 = sub_145CF80(*(_QWORD *)(a2 + 16), v8, 1, 0);
  v11 = *(_QWORD *)(a2 + 16);
  if ( v31 )
  {
    v12 = v32;
    v13 = sub_14806B0(v11, v35, v10, 0, 0);
  }
  else
  {
    v39 = v10;
    v38 = v35;
    v37 = 0x200000002LL;
    v30 = v10;
    v36 = &v38;
    v20 = sub_147DD40(v11, (__int64 *)&v36, 0, 0, a4, a5);
    v21 = v30;
    v12 = (__int64)v20;
    if ( v36 != &v38 )
    {
      _libc_free((unsigned __int64)v36);
      v21 = v30;
    }
    v22 = *(_QWORD *)(a2 + 16);
    v39 = v21;
    v38 = v32;
    v37 = 0x200000002LL;
    v36 = &v38;
    v35 = (__int64)sub_147DD40(v22, (__int64 *)&v36, 0, 0, a4, a5);
    if ( v36 != &v38 )
      _libc_free((unsigned __int64)v36);
    v13 = v32;
  }
  v14 = *(_QWORD *)(a2 + 16);
  v15 = *(_QWORD *)(a2 + 88);
  if ( a3 )
  {
    v33 = v13;
    if ( (unsigned __int8)sub_147A340(v14, 0x29u, v15, v12) )
    {
      v6 = sub_147A340(*(_QWORD *)(a2 + 16), 0x28u, v33, *(_QWORD *)(a2 + 96));
      if ( v6 )
      {
LABEL_9:
        *(_BYTE *)(a1 + 32) = 1;
        *(_BYTE *)(a1 + 8) = 0;
        *(_BYTE *)(a1 + 24) = 0;
        return a1;
      }
      goto LABEL_19;
    }
    v25 = *(_QWORD **)(a2 + 16);
    v26 = sub_1480950(v25, v35, *(_QWORD *)(a2 + 88), a4, a5);
    v29 = sub_147A9C0(v25, v12, v26, a4, a5);
    if ( !(unsigned __int8)sub_147A340(*(_QWORD *)(a2 + 16), 0x28u, v33, *(_QWORD *)(a2 + 96)) )
    {
LABEL_19:
      v23 = *(_QWORD **)(a2 + 16);
      v24 = sub_1480950(v23, v35, *(_QWORD *)(a2 + 96), a4, a5);
      v19 = sub_147A9C0(v23, v12, v24, a4, a5);
      goto LABEL_20;
    }
LABEL_24:
    *(_BYTE *)(a1 + 32) = 1;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = v29;
    *(_BYTE *)(a1 + 24) = 0;
    return a1;
  }
  v34 = v13;
  if ( (unsigned __int8)sub_147A340(v14, 0x25u, v15, v12) )
  {
    v6 = sub_147A340(*(_QWORD *)(a2 + 16), 0x24u, v34, *(_QWORD *)(a2 + 96));
    if ( v6 )
      goto LABEL_9;
    v16 = *(_QWORD *)(a2 + 96);
  }
  else
  {
    v27 = *(_QWORD **)(a2 + 16);
    v28 = sub_1481BD0(v27, v35, *(_QWORD *)(a2 + 88), a4, a5);
    v29 = sub_14819D0(v27, v12, v28, a4, a5);
    if ( (unsigned __int8)sub_147A340(*(_QWORD *)(a2 + 16), 0x24u, v34, *(_QWORD *)(a2 + 96)) )
      goto LABEL_24;
    v16 = *(_QWORD *)(a2 + 96);
    v6 = 1;
  }
  v17 = *(_QWORD **)(a2 + 16);
  v18 = sub_1481BD0(v17, v35, v16, a4, a5);
  v19 = sub_14819D0(v17, v12, v18, a4, a5);
LABEL_20:
  *(_BYTE *)(a1 + 32) = 1;
  *(_BYTE *)(a1 + 8) = v6;
  if ( v6 )
    *(_QWORD *)a1 = v29;
  *(_BYTE *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 16) = v19;
  return a1;
}
