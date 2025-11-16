// Function: sub_21BEE70
// Address: 0x21bee70
//
unsigned __int64 __fastcall sub_21BEE70(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 *v6; // rax
  unsigned int v7; // edx
  __int64 v8; // r14
  __int64 v9; // r15
  unsigned int v10; // eax
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rsi
  _QWORD *v14; // r10
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int64 result; // rax
  char *v24; // rax
  __int64 v25; // rsi
  __int64 v26; // r14
  unsigned __int8 v27; // r8
  const void **v28; // r15
  __int128 v29; // [rsp-10h] [rbp-70h]
  unsigned __int64 v30; // [rsp-10h] [rbp-70h]
  __int64 v31; // [rsp+0h] [rbp-60h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  unsigned int v33; // [rsp+14h] [rbp-4Ch]
  _QWORD *v34; // [rsp+18h] [rbp-48h]
  unsigned __int8 v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+20h] [rbp-40h] BYREF
  int v37; // [rsp+28h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 32);
  v7 = *(_DWORD *)(a2 + 84);
  v8 = *v6;
  v9 = v6[1];
  v10 = *(_DWORD *)(a2 + 88);
  if ( !v10 )
  {
    if ( v7 == 4 )
    {
      v11 = 4846;
      if ( *(_BYTE *)(a1[58] + 936) )
        v11 = 4847 - ((unsigned int)((unsigned __int8)sub_21BE280((__int64)a1) == 0) - 1);
      goto LABEL_8;
    }
    if ( v7 > 4 )
    {
      if ( v7 == 5 )
      {
        v11 = 4852;
        if ( *(_BYTE *)(a1[58] + 936) )
          v11 = 4853 - ((unsigned int)((unsigned __int8)sub_21BE280((__int64)a1) == 0) - 1);
        goto LABEL_8;
      }
      goto LABEL_38;
    }
    if ( v7 != 1 )
    {
      if ( v7 == 3 )
      {
        v11 = 4855;
        if ( *(_BYTE *)(a1[58] + 936) )
          v11 = (unsigned int)(*(_DWORD *)(a1[60] + 82304) == 32) + 4856;
        goto LABEL_8;
      }
LABEL_38:
      sub_16BD130("Bad address space in addrspacecast", 1u);
    }
    v11 = 4849 - ((unsigned int)(*(_BYTE *)(a1[58] + 936) == 0) - 1);
LABEL_8:
    v12 = *(_QWORD *)(a2 + 40);
    v13 = *(_QWORD *)(a2 + 72);
    v14 = (_QWORD *)a1[34];
    v15 = *(_QWORD *)(v12 + 8);
    v16 = **(unsigned __int8 **)(a2 + 40);
    v36 = v13;
    if ( v13 )
    {
      v31 = v16;
      v32 = v15;
      v33 = v11;
      v34 = v14;
      sub_1623A60((__int64)&v36, v13, 2);
      v16 = v31;
      v15 = v32;
      v11 = v33;
      v14 = v34;
    }
    *((_QWORD *)&v29 + 1) = v9;
    *(_QWORD *)&v29 = v8;
    v37 = *(_DWORD *)(a2 + 64);
    v17 = sub_1D2CC80(v14, v11, (__int64)&v36, v16, v15, v11, v29);
    goto LABEL_11;
  }
  if ( !v7 )
  {
    if ( v10 == 4 )
    {
      v11 = 4858;
      if ( *(_BYTE *)(a1[58] + 936) )
        v11 = (unsigned int)((unsigned __int8)sub_21BE280((__int64)a1) == 0) + 4859;
      goto LABEL_8;
    }
    if ( v10 > 4 )
    {
      if ( v10 == 5 )
      {
        v11 = 4864;
        if ( *(_BYTE *)(a1[58] + 936) )
          v11 = (unsigned int)((unsigned __int8)sub_21BE280((__int64)a1) == 0) + 4865;
        goto LABEL_8;
      }
      if ( v10 != 101 )
        goto LABEL_38;
      v11 = 4883 - ((unsigned int)(*(_BYTE *)(a1[58] + 936) == 0) - 1);
    }
    else
    {
      if ( v10 != 1 )
      {
        if ( v10 == 3 )
        {
          v11 = 4867;
          if ( *(_BYTE *)(a1[58] + 936) )
            v11 = (unsigned int)(*(_DWORD *)(a1[60] + 82304) != 32) + 4868;
          goto LABEL_8;
        }
        goto LABEL_38;
      }
      v11 = *(_BYTE *)(a1[58] + 936) == 0 ? 4861 : 4863;
    }
    goto LABEL_8;
  }
  v24 = *(char **)(a2 + 40);
  v25 = *(_QWORD *)(a2 + 72);
  v26 = a1[34];
  v27 = *v24;
  v28 = (const void **)*((_QWORD *)v24 + 1);
  v36 = v25;
  if ( v25 )
  {
    v35 = v27;
    sub_1623A60((__int64)&v36, v25, 2);
    v27 = v35;
  }
  v37 = *(_DWORD *)(a2 + 64);
  v17 = sub_1D38BB0(v26, 0, (__int64)&v36, v27, v28, 1, a3, a4, a5, 0);
LABEL_11:
  v18 = v17;
  sub_1D444E0(a1[34], a2, v17);
  sub_1D49010(v18);
  sub_1D2DC70((const __m128i *)a1[34], a2, v19, v20, v21, v22);
  result = v30;
  if ( v36 )
    return sub_161E7C0((__int64)&v36, v36);
  return result;
}
