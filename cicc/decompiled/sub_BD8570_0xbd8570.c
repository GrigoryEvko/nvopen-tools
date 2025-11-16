// Function: sub_BD8570
// Address: 0xbd8570
//
__int64 __fastcall sub_BD8570(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  unsigned int v6; // edx
  unsigned __int64 v7; // r14
  _BYTE *v8; // rax
  _BYTE *i; // rdx
  __int64 v10; // rsi
  size_t v11; // rax
  size_t v12; // r15
  unsigned int v13; // eax
  unsigned int v14; // r8d
  _QWORD *v15; // rcx
  __int64 v16; // rax
  unsigned int v17; // r8d
  _QWORD *v18; // rcx
  _QWORD *v19; // r12
  __int64 *v20; // rax
  __int64 v21; // r12
  __int64 *v22; // rax
  unsigned int v24; // [rsp+Ch] [rbp-B4h]
  bool v26; // [rsp+18h] [rbp-A8h]
  _QWORD *v27; // [rsp+18h] [rbp-A8h]
  bool v28; // [rsp+20h] [rbp-A0h]
  unsigned int v29; // [rsp+20h] [rbp-A0h]
  void *src; // [rsp+28h] [rbp-98h]
  void *v31; // [rsp+30h] [rbp-90h]
  __int64 v32; // [rsp+38h] [rbp-88h]
  _QWORD v33[3]; // [rsp+50h] [rbp-70h] BYREF
  _BYTE *v34; // [rsp+68h] [rbp-58h]
  _BYTE *v35; // [rsp+70h] [rbp-50h]
  __int64 v36; // [rsp+78h] [rbp-48h]
  _QWORD *v37; // [rsp+80h] [rbp-40h]

  v4 = a3[1];
  v24 = v4;
  if ( *(_BYTE *)a2 > 3u )
  {
    v26 = 0;
    v28 = 0;
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 40);
    if ( v5 )
    {
      v6 = *(_DWORD *)(v5 + 264) - 42;
      v26 = v6 < 2;
      v28 = v6 >= 2;
    }
    else
    {
      v26 = 0;
      v28 = 1;
    }
  }
  v7 = (unsigned int)v4;
  while ( 1 )
  {
    if ( v7 != v4 )
    {
      if ( v7 >= v4 )
      {
        if ( v7 > a3[2] )
        {
          sub_C8D290(a3, a3 + 3, v7, 1);
          v4 = a3[1];
        }
        v8 = (_BYTE *)(*a3 + v4);
        for ( i = (_BYTE *)(v7 + *a3); i != v8; ++v8 )
        {
          if ( v8 )
            *v8 = 0;
        }
      }
      a3[1] = v7;
    }
    v36 = 0x100000000LL;
    v33[1] = 2;
    v33[2] = 0;
    v34 = 0;
    v35 = 0;
    v33[0] = &unk_49DD288;
    v37 = a3;
    sub_CB5980(v33, 0, 0, 0);
    if ( v28 )
    {
      if ( v34 == v35 )
        sub_CB6200(v33, ".", 1);
      else
        *v35++ = 46;
    }
    else if ( v26 )
    {
      if ( v34 == v35 )
        sub_CB6200(v33, "$", 1);
      else
        *v35++ = 36;
    }
    v10 = (unsigned int)(*(_DWORD *)(a1 + 28) + 1);
    *(_DWORD *)(a1 + 28) = v10;
    sub_CB59D0(v33, v10);
    v11 = *(int *)(a1 + 24);
    v12 = a3[1];
    if ( (v11 & 0x80000000) == 0LL && v11 < v12 )
    {
      v24 -= v12 - v11;
      v7 = v24;
      goto LABEL_20;
    }
    v32 = a3[1];
    v31 = (void *)*a3;
    src = (void *)*a3;
    v13 = sub_C92610(*a3, v32);
    v14 = sub_C92740(a1, v31, v32, v13);
    v15 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v14);
    if ( !*v15 )
      goto LABEL_23;
    if ( *v15 == -8 )
      break;
LABEL_20:
    v33[0] = &unk_49DD388;
    sub_CB5840(v33);
    v4 = a3[1];
  }
  --*(_DWORD *)(a1 + 16);
LABEL_23:
  v27 = v15;
  v29 = v14;
  v16 = sub_C7D670(v12 + 17, 8);
  v17 = v29;
  v18 = v27;
  v19 = (_QWORD *)v16;
  if ( v12 )
  {
    memcpy((void *)(v16 + 16), src, v12);
    v17 = v29;
    v18 = v27;
  }
  *((_BYTE *)v19 + v12 + 16) = 0;
  *v19 = v12;
  v19[1] = a2;
  *v18 = v19;
  ++*(_DWORD *)(a1 + 12);
  v20 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0(a1, v17));
  v21 = *v20;
  if ( *v20 == -8 || !v21 )
  {
    v22 = v20 + 1;
    do
    {
      do
        v21 = *v22++;
      while ( v21 == -8 );
    }
    while ( !v21 );
  }
  v33[0] = &unk_49DD388;
  sub_CB5840(v33);
  return v21;
}
