// Function: sub_2644B30
// Address: 0x2644b30
//
__int64 *__fastcall sub_2644B30(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool v6; // zf
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rax
  __m128i v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v21; // [rsp+18h] [rbp-148h]
  char *v22[4]; // [rsp+20h] [rbp-140h] BYREF
  __m128i v23[2]; // [rsp+40h] [rbp-120h] BYREF
  char v24; // [rsp+60h] [rbp-100h]
  char v25; // [rsp+61h] [rbp-FFh]
  __m128i v26[2]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v27; // [rsp+90h] [rbp-D0h]
  __m128i v28[3]; // [rsp+A0h] [rbp-C0h] BYREF
  __m128i v29[2]; // [rsp+D0h] [rbp-90h] BYREF
  __int16 v30; // [rsp+F0h] [rbp-70h]
  __m128i v31[2]; // [rsp+100h] [rbp-60h] BYREF
  __int16 v32; // [rsp+120h] [rbp-40h]

  v6 = *(_BYTE *)a2 == 0;
  v30 = 267;
  v29[0].m128i_i64[0] = a2 + 40;
  if ( v6 )
  {
    v27 = 257;
  }
  else
  {
    v26[0].m128i_i64[0] = (__int64)"Alloc";
    v27 = 259;
  }
  v25 = 1;
  v23[0].m128i_i64[0] = (__int64)"OrigId: ";
  v24 = 3;
  sub_9C6370(v28, v23, v26, a4, a5, (__int64)v26);
  sub_9C6370(v31, v28, v29, v7, v8, v9);
  sub_CA0F50(a1, (void **)v31);
  sub_2241520((unsigned __int64 *)a1, "\n");
  v10 = *(_QWORD *)(a2 + 8);
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 - 32);
    if ( v11 )
    {
      if ( *(_BYTE *)v11 )
      {
        v11 = 0;
      }
      else if ( *(_QWORD *)(v11 + 24) != *(_QWORD *)(v10 + 80) )
      {
        v11 = 0;
      }
    }
    v21 = *(_QWORD *)(a2 + 8);
    v26[0].m128i_i64[0] = (__int64)sub_BD5D20(v11);
    v27 = 261;
    v26[0].m128i_i64[1] = v12;
    v29[0].m128i_i64[0] = (__int64)" -> ";
    v30 = 259;
    v13 = sub_B43CB0(v21);
    v14.m128i_i64[0] = (__int64)sub_BD5D20(v13);
    v32 = 261;
    v31[0] = v14;
    sub_9C6370(v28, v31, v29, v15, v16, v17);
    sub_9C6370(v23, v28, v26, v18, v19, (__int64)v26);
    sub_CA0F50((__int64 *)v22, (void **)v23);
    sub_2241490((unsigned __int64 *)a1, v22[0], (size_t)v22[1]);
    sub_2240A30((unsigned __int64 *)v22);
  }
  else
  {
    sub_2241520((unsigned __int64 *)a1, "null call");
    if ( *(_BYTE *)(a2 + 1) )
      sub_2241520((unsigned __int64 *)a1, " (recursive)");
    else
      sub_2241520((unsigned __int64 *)a1, " (external)");
  }
  return a1;
}
