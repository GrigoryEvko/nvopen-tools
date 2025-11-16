// Function: sub_29B0ED0
// Address: 0x29b0ed0
//
__int64 __fastcall sub_29B0ED0(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  int v6; // esi
  __int64 v7; // r9
  __int64 v8; // r8
  int v9; // edi
  __int64 v10; // rcx
  unsigned int v11; // edx
  __int64 v12; // r10
  char v14; // dh
  __int64 *v15; // rsi
  char v16; // al
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rbx
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rsi
  int v25; // ecx
  unsigned int v26; // edx
  __int64 v27; // rdi
  int v28; // r11d
  int v29; // r8d
  unsigned __int64 v30; // rdx
  __int64 v31; // rdi
  _QWORD *v32; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v33[32]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v34; // [rsp+30h] [rbp-30h]

  v4 = a2[2];
  v32 = a2;
  if ( v4 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(v4 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v5 - 30) <= 0xAu )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        goto LABEL_13;
    }
    v6 = *(_DWORD *)(a1 + 80);
    v7 = *(_QWORD *)(a1 + 64);
    v8 = 0;
    v9 = v6 - 1;
LABEL_5:
    v10 = *(_QWORD *)(v5 + 40);
    if ( !v6 )
      goto LABEL_9;
    v11 = v9 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v12 = *(_QWORD *)(v7 + 8LL * v11);
    if ( v10 == v12 )
    {
LABEL_7:
      if ( !v8 )
      {
        v8 = v10;
        goto LABEL_9;
      }
      if ( v10 == v8 )
        goto LABEL_9;
    }
    else
    {
      v28 = 1;
      while ( v12 != -4096 )
      {
        v11 = v9 & (v28 + v11);
        v12 = *(_QWORD *)(v7 + 8LL * v11);
        if ( v10 == v12 )
          goto LABEL_7;
        ++v28;
      }
LABEL_9:
      while ( 1 )
      {
        v4 = *(_QWORD *)(v4 + 8);
        if ( !v4 )
          break;
        v5 = *(_QWORD *)(v4 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v5 - 30) <= 0xAu )
          goto LABEL_5;
      }
      if ( v8 )
        return v8;
    }
  }
LABEL_13:
  v34 = 257;
  v15 = (__int64 *)sub_AA4FF0((__int64)a2);
  v16 = 0;
  if ( v15 )
    v16 = v14;
  v17 = 1;
  BYTE1(v17) = v16;
  v18 = sub_AA8550(a2, v15, v17, (__int64)v33, 0);
  v19 = v32[2];
  while ( v19 )
  {
    v20 = *(_QWORD *)(v19 + 24);
    v21 = v19;
    v19 = *(_QWORD *)(v19 + 8);
    if ( (unsigned __int8)(*(_BYTE *)v20 - 30) <= 0xAu )
    {
      do
      {
        v21 = *(_QWORD *)(v21 + 8);
        if ( !v21 )
          break;
LABEL_19:
        ;
      }
      while ( (unsigned __int8)(**(_BYTE **)(v21 + 24) - 30) > 0xAu );
LABEL_20:
      v22 = *(_DWORD *)(a1 + 80);
      v23 = *(_QWORD *)(v20 + 40);
      v24 = *(_QWORD *)(a1 + 64);
      if ( v22 )
      {
        v25 = v22 - 1;
        v26 = (v22 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v27 = *(_QWORD *)(v24 + 8LL * v26);
        if ( v27 == v23 )
        {
LABEL_22:
          if ( !v21 )
            break;
LABEL_23:
          v20 = *(_QWORD *)(v21 + 24);
          v21 = *(_QWORD *)(v21 + 8);
          if ( v21 )
            goto LABEL_19;
          goto LABEL_20;
        }
        v29 = 1;
        while ( v27 != -4096 )
        {
          v26 = v25 & (v29 + v26);
          v27 = *(_QWORD *)(v24 + 8LL * v26);
          if ( v23 == v27 )
            goto LABEL_22;
          ++v29;
        }
      }
      v30 = *(_QWORD *)(v23 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v30 == v23 + 48 )
      {
        v31 = 0;
      }
      else
      {
        if ( !v30 )
          BUG();
        v31 = v30 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v30 - 24) - 30 >= 0xB )
          v31 = 0;
      }
      sub_BD2ED0(v31, (__int64)v32, v18);
      if ( !v21 )
        break;
      goto LABEL_23;
    }
  }
  sub_29B0C40(a1 + 56, (__int64 *)&v32);
  return (__int64)v32;
}
