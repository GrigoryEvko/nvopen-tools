// Function: sub_87C590
// Address: 0x87c590
//
__int64 __fastcall sub_87C590(__int64 a1, const char *a2, __int64 a3, int a4, int a5, _DWORD *a6, __int64 *a7)
{
  unsigned __int8 v9; // r14
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  char v15; // r14
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r15
  char v19; // r12
  char v20; // al
  __int64 v21; // rdx
  __int64 i; // rax
  __int64 v23; // rdx
  _QWORD *v25; // r13
  _QWORD *v26; // r14
  __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rdx
  _QWORD **v30; // rbx
  _QWORD *v31; // rdx
  __int64 v32; // rdi
  __int64 j; // r14
  _QWORD *v34; // r15
  _QWORD *v35; // rax
  __int64 k; // r9
  char v37; // al
  __int64 v38; // rdi
  __int64 v41; // [rsp+20h] [rbp-90h]
  __int64 v42; // [rsp+28h] [rbp-88h]
  __int64 v43; // [rsp+28h] [rbp-88h]
  _QWORD **v44; // [rsp+30h] [rbp-80h]
  int v45; // [rsp+38h] [rbp-78h]
  __int64 v46; // [rsp+38h] [rbp-78h]
  _BYTE v47[4]; // [rsp+44h] [rbp-6Ch] BYREF
  __int64 v48; // [rsp+48h] [rbp-68h] BYREF
  __int64 v49; // [rsp+50h] [rbp-60h] BYREF
  _BYTE v50[8]; // [rsp+58h] [rbp-58h] BYREF
  _BYTE v51[80]; // [rsp+60h] [rbp-50h] BYREF

  *a6 = 0;
  v41 = *(_QWORD *)(a1 + 88);
  v9 = 2 * (*(_BYTE *)(v41 + 176) != 1) + 2;
  if ( a2 && (v11 = sub_7D3790(v9, a2)) != 0 )
  {
    *a7 = v11;
  }
  else
  {
    v11 = sub_7D3810(v9);
    *a7 = v11;
    if ( !v11 )
      return v11;
  }
  v12 = *(_QWORD *)(v41 + 152);
  v13 = *(unsigned __int8 *)(v12 + 140);
  if ( !a4 )
  {
    if ( (_BYTE)v13 == 12 )
    {
      v29 = *(_QWORD *)(v41 + 152);
      do
        v29 = *(_QWORD *)(v29 + 160);
      while ( *(_BYTE *)(v29 + 140) == 12 );
      v14 = *(_QWORD *)(v29 + 168);
      if ( (*(_BYTE *)(v14 + 16) & 1) != 0 )
        goto LABEL_6;
    }
    else
    {
      v14 = *(_QWORD *)(v12 + 168);
      if ( (*(_BYTE *)(v14 + 16) & 1) != 0 )
        goto LABEL_8;
    }
    v30 = *(_QWORD ***)v14;
    v31 = **(_QWORD ***)v14;
    if ( v31 )
    {
      if ( !dword_4D04818 )
        goto LABEL_5;
      v32 = v31[1];
      if ( v32 != unk_4F06C60 )
      {
        if ( !(unsigned int)sub_8D97D0(v32, unk_4F06C60, 0, v13, v10) )
        {
LABEL_45:
          v12 = *(_QWORD *)(v41 + 152);
          LOBYTE(v13) = *(_BYTE *)(v12 + 140);
          goto LABEL_5;
        }
        v31 = *v30;
      }
      if ( *v31 )
        goto LABEL_45;
    }
    return sub_87C270(v11, a3, a6);
  }
LABEL_5:
  if ( (_BYTE)v13 == 12 )
  {
    do
LABEL_6:
      v12 = *(_QWORD *)(v12 + 160);
    while ( *(_BYTE *)(v12 + 140) == 12 );
  }
  v14 = *(_QWORD *)(v12 + 168);
LABEL_8:
  v15 = *(_BYTE *)(v14 + 16) & 1;
  v44 = *(_QWORD ***)v14;
  v45 = 0;
  v18 = sub_82C1B0(v11, 0, 0, (__int64)v51);
  if ( !v18 )
    return 0;
  v42 = 0;
  v19 = v15;
  while ( 1 )
  {
    v20 = *(_BYTE *)(v18 + 80);
    v21 = v18;
    if ( v20 == 16 )
      break;
LABEL_11:
    if ( v20 == 20 )
    {
      v45 = 1;
    }
    else
    {
      for ( i = *(_QWORD *)(*(_QWORD *)(v21 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v23 = *(_QWORD *)(i + 168);
      if ( v19 == (*(_BYTE *)(v23 + 16) & 1) )
      {
        v25 = **(_QWORD ***)v23;
        v26 = *v44;
        while ( v25 )
        {
          if ( !v26 )
            goto LABEL_15;
          v27 = v25[1];
          v28 = v26[1];
          if ( v27 != v28 && !(unsigned int)sub_8D97D0(v27, v28, 0, v16, v17) )
            goto LABEL_15;
          v25 = (_QWORD *)*v25;
          v26 = (_QWORD *)*v26;
        }
        if ( !v26 )
        {
          if ( v42 )
          {
            *a6 = 1;
            return 0;
          }
          v42 = v18;
        }
      }
    }
LABEL_15:
    v18 = sub_82C230(v51);
    if ( !v18 )
    {
      v11 = v42;
      goto LABEL_17;
    }
  }
  if ( (*(_BYTE *)(v18 + 82) & 4) == 0 )
  {
    v21 = **(_QWORD **)(v18 + 88);
    v20 = *(_BYTE *)(v21 + 80);
    if ( v20 == 24 )
    {
      v21 = *(_QWORD *)(v21 + 88);
      v20 = *(_BYTE *)(v21 + 80);
    }
    goto LABEL_11;
  }
  v11 = v42;
  *a6 = 1;
LABEL_17:
  if ( v45 )
  {
    v48 = 0;
    if ( !v11 )
    {
      if ( *a6 )
        return 0;
      for ( j = *(_QWORD *)(v41 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      v46 = *(_QWORD *)(j + 160);
      *(_QWORD *)(j + 160) = sub_72CBE0();
      v34 = v44[1];
      v35 = (_QWORD *)sub_72CBE0();
      v44[1] = (_QWORD *)sub_72D2E0(v35);
      for ( k = sub_82C1B0(*a7, 0, 0, (__int64)v51); k; k = sub_82C230(v51) )
      {
        v37 = *(_BYTE *)(k + 80);
        v38 = k;
        if ( v37 == 16 )
        {
          v38 = **(_QWORD **)(k + 88);
          v37 = *(_BYTE *)(v38 + 80);
        }
        if ( v37 == 24 )
        {
          v38 = *(_QWORD *)(v38 + 88);
          v37 = *(_BYTE *)(v38 + 80);
        }
        if ( v37 == 20 )
        {
          v43 = k;
          if ( (unsigned int)sub_8B8060(v38, j, 0, 1, 0, k) )
            sub_8B5FF0(&v48, v43, 0);
        }
      }
      if ( v48 )
      {
        sub_893120(v48, 0, &v49, v50, a6, 0);
        if ( !*a6 )
        {
          v11 = v49;
          if ( !a5 )
            v11 = sub_8B7F20(v49, j, 0, 0, 1, 0, 0, (__int64)v47);
        }
      }
      *(_QWORD *)(j + 160) = v46;
      v44[1] = v34;
    }
  }
  return v11;
}
