// Function: sub_1A4F870
// Address: 0x1a4f870
//
__int64 __fastcall sub_1A4F870(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 *v7; // rbx
  bool v8; // zf
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  char *v13; // r14
  char *v14; // rbx
  __int64 v15; // r12
  char v16; // r8
  unsigned int v17; // r13d
  __int64 v18; // rdi
  int v19; // ecx
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 v22; // r10
  unsigned int v23; // r10d
  _QWORD *i; // rax
  __int64 v25; // rsi
  unsigned int v26; // edx
  __int64 v27; // rax
  __int64 v28; // r11
  _QWORD *v29; // rdx
  unsigned int j; // eax
  int v31; // ecx
  __int64 v32; // rax
  int v33; // eax
  int v34; // esi
  int v35; // eax
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 *v38; // rbx
  __int64 v39; // rcx
  __int64 v40; // rax
  int v41; // [rsp+Ch] [rbp-74h]
  char *v42; // [rsp+10h] [rbp-70h]
  __int64 v43; // [rsp+18h] [rbp-68h]
  __int64 *v45; // [rsp+28h] [rbp-58h]
  __int64 v46[7]; // [rsp+48h] [rbp-38h] BYREF

  result = (char *)a2 - (char *)a1;
  v45 = a2;
  v43 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
LABEL_38:
    v36 = result >> 3;
    v37 = ((result >> 3) - 2) >> 1;
    sub_1A4F6D0((__int64)a1, v37, result >> 3, a1[v37], a4);
    do
    {
      --v37;
      sub_1A4F6D0((__int64)a1, v37, v36, a1[v37], a4);
    }
    while ( v37 );
    v38 = v45;
    do
    {
      v39 = *--v38;
      *v38 = *a1;
      result = sub_1A4F6D0((__int64)a1, 0, v38 - a1, v39, a4);
    }
    while ( (char *)v38 - (char *)a1 > 8 );
    return result;
  }
  v42 = (char *)(a1 + 1);
  while ( 2 )
  {
    v6 = a1[1];
    v7 = &a1[result >> 4];
    --v43;
    v46[0] = a4;
    v8 = !sub_1A4F560(v46, v6, *v7);
    v9 = *(v45 - 1);
    if ( v8 )
    {
      if ( !sub_1A4F560(v46, a1[1], v9) )
      {
        v8 = !sub_1A4F560(v46, *v7, *(v45 - 1));
        v10 = *a1;
        if ( !v8 )
        {
          *a1 = *(v45 - 1);
          *(v45 - 1) = v10;
          v11 = *a1;
          v12 = a1[1];
          goto LABEL_8;
        }
        goto LABEL_7;
      }
LABEL_44:
      v12 = *a1;
      v11 = a1[1];
      a1[1] = *a1;
      *a1 = v11;
      goto LABEL_8;
    }
    if ( !sub_1A4F560(v46, *v7, v9) )
    {
      if ( sub_1A4F560(v46, a1[1], *(v45 - 1)) )
      {
        v40 = *a1;
        *a1 = *(v45 - 1);
        *(v45 - 1) = v40;
        v11 = *a1;
        v12 = a1[1];
        goto LABEL_8;
      }
      goto LABEL_44;
    }
    v10 = *a1;
LABEL_7:
    *a1 = *v7;
    *v7 = v10;
    v11 = *a1;
    v12 = a1[1];
LABEL_8:
    v13 = v42;
    v14 = (char *)v45;
    v46[0] = a4;
    while ( 1 )
    {
      if ( sub_1A4F560(v46, v12, v11) )
        goto LABEL_27;
      v14 -= 8;
      v15 = *a1;
      v16 = *(_BYTE *)(v46[0] + 8) & 1;
      v17 = ((unsigned int)*a1 >> 9) ^ ((unsigned int)*a1 >> 4);
      if ( v16 )
      {
LABEL_11:
        v18 = v46[0] + 16;
        v19 = 15;
        goto LABEL_12;
      }
      while ( 1 )
      {
        v31 = *(_DWORD *)(v46[0] + 24);
        v18 = *(_QWORD *)(v46[0] + 16);
        if ( !v31 )
          BUG();
        v19 = v31 - 1;
LABEL_12:
        v20 = v19 & v17;
        v21 = v18 + 16LL * (v19 & v17);
        v22 = *(_QWORD *)v21;
        if ( v15 != *(_QWORD *)v21 )
        {
          v33 = 1;
          while ( v22 != -8 )
          {
            v34 = v33 + 1;
            v20 = v19 & (v33 + v20);
            v21 = v18 + 16LL * v20;
            v22 = *(_QWORD *)v21;
            if ( v15 == *(_QWORD *)v21 )
              goto LABEL_13;
            v33 = v34;
          }
LABEL_49:
          BUG();
        }
LABEL_13:
        v23 = 1;
        for ( i = **(_QWORD ***)(v21 + 8); i; ++v23 )
          i = (_QWORD *)*i;
        if ( !v16 && !*(_DWORD *)(v46[0] + 24) )
          goto LABEL_49;
        v25 = *(_QWORD *)v14;
        v26 = v19 & (((unsigned int)*(_QWORD *)v14 >> 9) ^ ((unsigned int)*(_QWORD *)v14 >> 4));
        v27 = v18 + 16LL * v26;
        v28 = *(_QWORD *)v27;
        if ( *(_QWORD *)v14 != *(_QWORD *)v27 )
        {
          v35 = 1;
          while ( v28 != -8 )
          {
            v26 = v19 & (v35 + v26);
            v41 = v35 + 1;
            v27 = v18 + 16LL * v26;
            v28 = *(_QWORD *)v27;
            if ( v25 == *(_QWORD *)v27 )
              goto LABEL_18;
            v35 = v41;
          }
          goto LABEL_49;
        }
LABEL_18:
        v29 = **(_QWORD ***)(v27 + 8);
        for ( j = 1; v29; ++j )
          v29 = (_QWORD *)*v29;
        if ( j <= v23 )
          break;
        v14 -= 8;
        if ( v16 )
          goto LABEL_11;
      }
      if ( v13 >= v14 )
        break;
      v32 = *(_QWORD *)v13;
      *(_QWORD *)v13 = v25;
      *(_QWORD *)v14 = v32;
LABEL_27:
      v12 = *((_QWORD *)v13 + 1);
      v13 += 8;
      v11 = *a1;
    }
    sub_1A4F870(v13, v45, v43, a4);
    result = v13 - (char *)a1;
    if ( v13 - (char *)a1 > 128 )
    {
      v45 = (__int64 *)v13;
      if ( !v43 )
        goto LABEL_38;
      continue;
    }
    return result;
  }
}
