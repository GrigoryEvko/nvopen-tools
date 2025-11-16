// Function: sub_7EBD50
// Address: 0x7ebd50
//
__int64 __fastcall sub_7EBD50(__int64 *a1)
{
  _QWORD *v1; // r12
  __int64 i; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // eax
  __int64 v16; // r9
  _QWORD *v17; // r14
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // r15
  char v21; // bl
  const __m128i *v22; // r8
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+18h] [rbp-48h] BYREF
  const __m128i *v26; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v27[7]; // [rsp+28h] [rbp-38h] BYREF

  v1 = (_QWORD *)a1[9];
  v26 = 0;
  v27[0] = 0;
  for ( i = sub_8D46C0(*v1); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v6 = *(_QWORD *)(i + 168);
  v7 = v1[2];
  v8 = *(unsigned __int8 *)(v6 + 16);
  v9 = *(_QWORD *)(v7 + 16);
  v25 = v7;
  if ( (v8 & 0x40) != 0 )
  {
    v10 = v9;
    if ( (v8 & 0x80u) == 0LL )
    {
      v25 = v9;
      v10 = v7;
      v7 = v9;
    }
    v9 = *(_QWORD *)(v9 + 16);
    *(_QWORD *)(v10 + 16) = 0;
    v1[2] = 0;
    *(_QWORD *)(v7 + 16) = 0;
    if ( (unsigned int)sub_731800(v9, 0, v8, v3, v4, v5) )
    {
      v17 = sub_7EBCC0((_QWORD **)v1, &v25, 1u, (__int64)v27, &v26, v14);
    }
    else
    {
      v15 = sub_731770(v10, 0, v11, v12, v13, v14);
      v17 = sub_7EBCC0((_QWORD **)v1, &v25, v15 != 0, (__int64)v27, &v26, v16);
    }
    v18 = v25;
    v17[2] = v10;
    *(_QWORD *)(v10 + 16) = v18;
  }
  else
  {
    v1[2] = 0;
    *(_QWORD *)(v7 + 16) = 0;
    if ( (unsigned int)sub_731800(v9, 0, v8, v3, v4, v5) )
      v17 = sub_7EBCC0((_QWORD **)v1, &v25, 1u, (__int64)v27, &v26, v19);
    else
      v17 = sub_7EBCC0((_QWORD **)v1, &v25, 0, (__int64)v27, &v26, v19);
    v18 = v25;
    v17[2] = v25;
  }
  *(_QWORD *)(v18 + 16) = v9;
  v20 = *a1;
  v24 = v1[2];
  v21 = *((_BYTE *)v1 + 25) >> 2;
  sub_7266C0((__int64)v1, 1);
  v1[2] = v24;
  *((_BYTE *)v1 + 25) = *((_BYTE *)v1 + 25) & 0xFB | (4 * (v21 & 1));
  sub_73D8E0((__int64)v1, 0x69u, v20, 0, (__int64)v17);
  v22 = v26;
  if ( !v26 )
    return sub_730620((__int64)a1, (const __m128i *)v1);
  v26[1].m128i_i64[0] = (__int64)v1;
  return sub_73D8E0((__int64)a1, 0x5Bu, *a1, 0, (__int64)v22);
}
