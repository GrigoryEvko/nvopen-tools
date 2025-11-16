// Function: sub_32260F0
// Address: 0x32260f0
//
void __fastcall sub_32260F0(__int64 a1, unsigned __int8 a2, _BYTE *a3)
{
  __int64 v4; // r12
  void (__fastcall *v5)(__int64, char, void **, __int64, __int64, __int64); // r15
  const char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  char v10; // cl
  char *v11; // rdi
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  __m128i **v15; // r13
  __int64 v16; // [rsp+8h] [rbp-B8h]
  __m128i v17; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v18; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v19[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v20; // [rsp+50h] [rbp-70h]
  const char *v21; // [rsp+60h] [rbp-60h] BYREF
  __int64 v22; // [rsp+68h] [rbp-58h]
  const char *v23; // [rsp+70h] [rbp-50h]
  __int64 v24; // [rsp+78h] [rbp-48h]
  __int16 v25; // [rsp+80h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 120) )
    v4 = *(_QWORD *)(a1 + 104) + 80LL;
  else
    v4 = *(_QWORD *)(a1 + 112);
  v5 = **(void (__fastcall ***)(__int64, char, void **, __int64, __int64, __int64))v4;
  if ( a3 )
  {
    v6 = sub_E06E20(a2);
    if ( *a3 )
    {
      v19[0] = a3;
      v19[2] = " ";
      v10 = 2;
      v20 = 771;
      v11 = (char *)v19;
    }
    else
    {
      v11 = " ";
      v19[0] = " ";
      v20 = 259;
      v16 = v19[1];
      v10 = 3;
    }
    v21 = v11;
    v23 = v6;
    v22 = v16;
    v24 = v7;
    LOBYTE(v25) = v10;
    HIBYTE(v25) = 5;
    if ( v5 == sub_3225B70 )
      goto LABEL_7;
LABEL_14:
    ((void (__fastcall *)(__int64, _QWORD, const char **))v5)(v4, a2, &v21);
    return;
  }
  v25 = 261;
  v21 = sub_E06E20(a2);
  v22 = v14;
  if ( v5 != sub_3225B70 )
    goto LABEL_14;
LABEL_7:
  v12 = *(_QWORD **)(v4 + 8);
  v13 = v12[1];
  if ( (unsigned __int64)(v13 + 1) > v12[2] )
  {
    sub_C8D290(*(_QWORD *)(v4 + 8), v12 + 3, v13 + 1, 1u, v8, v9);
    v13 = v12[1];
  }
  *(_BYTE *)(*v12 + v13) = a2;
  ++v12[1];
  if ( *(_BYTE *)(v4 + 24) )
  {
    v15 = *(__m128i ***)(v4 + 16);
    sub_CA0F50(v17.m128i_i64, (void **)&v21);
    sub_3225850(v15, &v17);
    if ( (__int64 *)v17.m128i_i64[0] != &v18 )
      j_j___libc_free_0(v17.m128i_u64[0]);
  }
}
