// Function: sub_E5E620
// Address: 0xe5e620
//
__int64 __fastcall sub_E5E620(_QWORD *a1, __int64 a2)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rbx
  char v8; // al
  char v9; // r15
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  __int64 v12; // rdi
  char *v13; // rax
  _BYTE *v14; // rbx
  char v15; // dl
  unsigned __int64 i; // rax
  char v17; // r15
  __int64 v18; // rdx
  unsigned int j; // ecx
  char v20; // r14
  char *v21; // rax
  char v22; // si
  unsigned int v23; // r9d
  char v24; // al
  unsigned int v25; // r15d
  char v26; // bl
  char *v27; // rax
  char *v28; // rax
  unsigned __int64 v30; // r14
  unsigned int v31; // r15d
  char v32; // si
  char *v33; // rax
  unsigned int v34; // ebx
  char *v35; // rax
  char *v36; // rax
  __int16 v37; // ax
  unsigned int v38; // [rsp+8h] [rbp-98h]
  unsigned int v39; // [rsp+10h] [rbp-90h]
  char v40; // [rsp+10h] [rbp-90h]
  __int64 v41; // [rsp+18h] [rbp-88h]
  unsigned int v42; // [rsp+18h] [rbp-88h]
  unsigned __int64 v43; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v44[2]; // [rsp+30h] [rbp-70h] BYREF
  const char *v45; // [rsp+40h] [rbp-60h]
  unsigned __int64 v46; // [rsp+48h] [rbp-58h]
  char *v47; // [rsp+50h] [rbp-50h]
  __int64 v48; // [rsp+58h] [rbp-48h]
  __int64 v49; // [rsp+60h] [rbp-40h]

  v4 = a2;
  v5 = *(_QWORD *)(a2 + 48);
  *(_DWORD *)(a2 + 80) = 0;
  v6 = *(_QWORD *)(a2 + 96);
  v38 = v5;
  LODWORD(v7) = v5;
  if ( *(_BYTE *)(a1[3] + 81LL) )
    v8 = sub_E81940(v6, &v43, a1);
  else
    v8 = sub_E81920(v6, &v43);
  v9 = v8;
  if ( !v8 )
  {
    v10 = a1[1];
    v11 = *(__int64 (**)())(*(_QWORD *)v10 + 168LL);
    if ( v11 == sub_E5B870
      || (v37 = ((__int64 (__fastcall *)(__int64, _QWORD *, __int64, unsigned __int64 *))v11)(v10, a1, a2, &v43),
          v9 = HIBYTE(v37),
          !(_BYTE)v37) )
    {
      v12 = *a1;
      v13 = ".s";
      if ( !*(_BYTE *)(a2 + 88) )
        v13 = ".u";
      v44[0] = v13;
      v45 = "leb128 expression is not absolute";
      LOWORD(v47) = 771;
      sub_E66880(v12, *(_QWORD *)(*(_QWORD *)(a2 + 96) + 8LL), v44);
      *(_QWORD *)(a2 + 96) = sub_E81A90(0, *a1, 0, 0);
    }
    v14 = v44;
    v15 = v43 & 0x7F;
    for ( i = v43 >> 7; i; i >>= 7 )
    {
      *v14++ = v15 | 0x80;
      v15 = i & 0x7F;
    }
    *v14 = v15;
    v7 = v14 + 1 - (_BYTE *)v44;
    if ( v38 >= (unsigned int)v7 )
      LODWORD(v7) = v38;
    if ( v9 )
      v43 = 0;
  }
  v48 = 0x100000000LL;
  *(_QWORD *)(a2 + 48) = 0;
  v44[1] = 2;
  v44[0] = &unk_49DD288;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v49 = a2 + 40;
  sub_CB5980((__int64)v44, 0, 0, 0);
  v17 = *(_BYTE *)(a2 + 88);
  if ( !v17 )
  {
    v30 = v43;
    v31 = 0;
    while ( 1 )
    {
      ++v31;
      v32 = v30 & 0x7F;
      v30 >>= 7;
      if ( v30 || v31 < (unsigned int)v7 )
        v32 |= 0x80u;
      v33 = v47;
      if ( (unsigned __int64)v47 < v46 )
      {
        ++v47;
        *v33 = v32;
        if ( !v30 )
          goto LABEL_42;
      }
      else
      {
        sub_CB5D20((__int64)v44, v32);
        if ( !v30 )
        {
LABEL_42:
          if ( v31 >= (unsigned int)v7 )
            goto LABEL_35;
          v34 = v7 - 1;
          if ( v31 < v34 )
          {
            do
            {
              while ( 1 )
              {
                v35 = v47;
                if ( (unsigned __int64)v47 >= v46 )
                  break;
                ++v31;
                ++v47;
                *v35 = 0x80;
                if ( v31 == v34 )
                  goto LABEL_48;
              }
              ++v31;
              sub_CB5D20((__int64)v44, 128);
            }
            while ( v31 != v34 );
          }
LABEL_48:
          v36 = v47;
          if ( (unsigned __int64)v47 >= v46 )
          {
            sub_CB5D20((__int64)v44, 0);
          }
          else
          {
            ++v47;
            *v36 = 0;
          }
          goto LABEL_35;
        }
      }
    }
  }
  v18 = v43;
  for ( j = 1; ; j = v23 )
  {
    v24 = v18;
    v22 = v18 & 0x7F;
    v18 >>= 7;
    if ( v18 )
    {
      if ( v18 != -1 || (v24 & 0x40) == 0 )
      {
        v20 = v17;
        goto LABEL_16;
      }
    }
    else
    {
      v20 = v17;
      if ( (v24 & 0x40) != 0 )
        goto LABEL_16;
    }
    v20 = 0;
    if ( j >= (unsigned int)v7 )
      break;
LABEL_16:
    v21 = v47;
    v22 |= 0x80u;
    if ( (unsigned __int64)v47 >= v46 )
      goto LABEL_25;
LABEL_17:
    v23 = j + 1;
    v47 = v21 + 1;
    *v21 = v22;
    if ( !v20 )
      goto LABEL_26;
LABEL_18:
    ;
  }
  v21 = v47;
  if ( (unsigned __int64)v47 < v46 )
    goto LABEL_17;
LABEL_25:
  v39 = j;
  v41 = v18;
  sub_CB5D20((__int64)v44, v22);
  j = v39;
  v18 = v41;
  v23 = v39 + 1;
  if ( v20 )
    goto LABEL_18;
LABEL_26:
  if ( j < (unsigned int)v7 )
  {
    v25 = v7 - 1;
    v26 = (v18 >> 63) | 0x80;
    v40 = (v18 >> 63) & 0x7F;
    if ( v25 > j )
    {
      while ( 1 )
      {
        v27 = v47;
        if ( (unsigned __int64)v47 < v46 )
        {
          ++v47;
          *v27 = v26;
          if ( v23 == v25 )
            break;
        }
        else
        {
          v42 = v23;
          sub_CB5D20((__int64)v44, v26);
          v23 = v42;
          if ( v42 == v25 )
            break;
        }
        ++v23;
      }
    }
    v28 = v47;
    if ( (unsigned __int64)v47 >= v46 )
    {
      sub_CB5D20((__int64)v44, v40);
    }
    else
    {
      ++v47;
      *v28 = v40;
    }
  }
LABEL_35:
  LOBYTE(v4) = *(_QWORD *)(v4 + 48) != v38;
  v44[0] = &unk_49DD388;
  sub_CB5840((__int64)v44);
  return (unsigned int)v4;
}
