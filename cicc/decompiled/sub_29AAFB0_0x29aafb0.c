// Function: sub_29AAFB0
// Address: 0x29aafb0
//
__int64 __fastcall sub_29AAFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // rdx
  int v13; // ecx
  __int64 v14; // rsi
  __int64 v15; // rdi
  int v16; // ecx
  int v17; // ebx
  unsigned int v18; // edx
  __int64 v19; // r9
  char v20; // bl
  char v21; // al
  __int64 v22; // r10
  unsigned __int8 *v23; // r11
  char v24; // cl
  char v25; // al
  __int64 v26; // r8
  __int64 v27; // rcx
  int v28; // ecx
  __int64 v29; // [rsp+0h] [rbp-40h]
  __int64 v30; // [rsp+0h] [rbp-40h]
  __int64 v31; // [rsp+0h] [rbp-40h]
  char v33; // [rsp+8h] [rbp-38h]
  char v34; // [rsp+8h] [rbp-38h]

  v6 = *(_QWORD *)(a4 + 16);
  if ( !v6 )
    goto LABEL_4;
  v9 = 0;
  v10 = 0;
  do
  {
    v11 = *(_QWORD *)(v6 + 24);
    if ( *(_BYTE *)v11 <= 0x1Cu )
      goto LABEL_4;
    if ( *(_BYTE *)v11 == 85 )
    {
      v27 = *(_QWORD *)(v11 - 32);
      if ( v27 )
      {
        if ( !*(_BYTE *)v27 && *(_QWORD *)(v27 + 24) == *(_QWORD *)(v11 + 80) && (*(_BYTE *)(v27 + 33) & 0x20) != 0 )
        {
          v28 = *(_DWORD *)(v27 + 36);
          if ( v28 == 211 )
          {
            if ( v10 )
              goto LABEL_4;
            v10 = *(_QWORD *)(v6 + 24);
            goto LABEL_9;
          }
          if ( v28 == 210 )
          {
            if ( v9 )
              goto LABEL_4;
            v9 = *(_QWORD *)(v6 + 24);
            goto LABEL_9;
          }
          if ( (unsigned int)(v28 - 68) <= 3 )
            goto LABEL_9;
        }
      }
    }
    v13 = *(_DWORD *)(a2 + 80);
    v14 = *(_QWORD *)(v11 + 40);
    v15 = *(_QWORD *)(a2 + 64);
    if ( !v13 )
      goto LABEL_4;
    v16 = v13 - 1;
    v17 = 1;
    v18 = v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v19 = *(_QWORD *)(v15 + 8LL * v18);
    if ( v19 != v14 )
    {
      while ( v19 != -4096 )
      {
        v18 = v16 & (v17 + v18);
        v19 = *(_QWORD *)(v15 + 8LL * v18);
        if ( v14 == v19 )
          goto LABEL_9;
        ++v17;
      }
      goto LABEL_4;
    }
LABEL_9:
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v6 );
  if ( v9 && v10 )
  {
    v29 = a2 + 56;
    v20 = sub_29AACC0(a2 + 56, v10) ^ 1;
    v21 = sub_29AACC0(v29, v9) ^ 1;
    v24 = v21;
    if ( !v20 )
    {
      if ( v21 )
      {
        v30 = a5;
        v33 = v21;
        v25 = sub_29AAEB0(v22, a3, v23);
        v24 = v33;
        v26 = v30;
        if ( !v25 )
          goto LABEL_4;
        goto LABEL_15;
      }
      goto LABEL_16;
    }
    v31 = a5;
    v34 = v21;
    if ( (unsigned __int8)sub_29AAEB0(v22, a3, v23) )
    {
      v24 = v34;
      v26 = v31;
      if ( v34 )
      {
LABEL_15:
        if ( !v26 )
          goto LABEL_4;
      }
LABEL_16:
      *(_BYTE *)a1 = v20;
      *(_BYTE *)(a1 + 1) = v24;
      *(_QWORD *)(a1 + 8) = v10;
      *(_QWORD *)(a1 + 16) = v9;
      return a1;
    }
  }
LABEL_4:
  *(_QWORD *)(a1 + 8) = 0;
  *(_WORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return a1;
}
