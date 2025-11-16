// Function: sub_163CAF0
// Address: 0x163caf0
//
__int64 __fastcall sub_163CAF0(__int64 a1)
{
  char v1; // r13
  __int64 v2; // rdi
  __int64 i; // rax
  __int64 v4; // rdx
  __int64 v5; // rsi
  __int64 *v6; // r12
  int v7; // r15d
  __int64 *v8; // r11
  unsigned int v9; // r9d
  __int64 *v10; // r8
  __int64 v11; // rcx
  unsigned int v12; // r12d
  int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  __int64 v18; // r12
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // r14
  __int64 v23; // r15
  bool v24; // zf
  int v25; // r9d
  __int64 *v26; // r8
  int v27; // r8d
  __int64 *v28; // rdi
  unsigned int v29; // r14d
  __int64 v30; // [rsp+10h] [rbp-160h] BYREF
  __int64 v31; // [rsp+18h] [rbp-158h]
  __int64 v32; // [rsp+20h] [rbp-150h]
  __int64 v33; // [rsp+28h] [rbp-148h]
  _QWORD *v34; // [rsp+30h] [rbp-140h] BYREF
  __int64 v35; // [rsp+38h] [rbp-138h]
  _QWORD v36[38]; // [rsp+40h] [rbp-130h] BYREF

  v1 = 1;
  v34 = v36;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v36[0] = a1;
  v2 = 0;
  v35 = 0x2000000001LL;
  LODWORD(i) = 1;
  while ( 1 )
  {
    v4 = 8LL * (unsigned int)i - 8;
    if ( !(_DWORD)i )
    {
LABEL_6:
      v12 = (v1 == 0) + 2;
      goto LABEL_7;
    }
    while ( 1 )
    {
      v5 = (unsigned int)v33;
      LODWORD(i) = i - 1;
      v6 = *(__int64 **)((char *)v34 + v4);
      LODWORD(v35) = i;
      if ( !(_DWORD)v33 )
      {
        ++v30;
        goto LABEL_48;
      }
      v7 = 1;
      v8 = 0;
      v9 = (v33 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v10 = (__int64 *)(v2 + 8LL * v9);
      v11 = *v10;
      if ( v6 != (__int64 *)*v10 )
        break;
LABEL_5:
      v4 -= 8;
      if ( !(_DWORD)i )
        goto LABEL_6;
    }
    while ( v11 != -8 )
    {
      if ( v8 || v11 != -16 )
        v10 = v8;
      v9 = (v33 - 1) & (v7 + v9);
      v11 = *(_QWORD *)(v2 + 8LL * v9);
      if ( v6 == (__int64 *)v11 )
        goto LABEL_5;
      ++v7;
      v8 = v10;
      v10 = (__int64 *)(v2 + 8LL * v9);
    }
    if ( !v8 )
      v8 = v10;
    ++v30;
    v14 = v32 + 1;
    if ( 4 * ((int)v32 + 1) < (unsigned int)(3 * v33) )
    {
      v15 = (unsigned int)(v33 - (v14 + HIDWORD(v32)));
      v16 = (unsigned int)v33 >> 3;
      if ( (unsigned int)v15 > (unsigned int)v16 )
        goto LABEL_19;
      sub_13B3B90((__int64)&v30, v33);
      if ( (_DWORD)v33 )
      {
        v16 = (unsigned int)(v33 - 1);
        v5 = v31;
        v27 = 1;
        v28 = 0;
        v29 = v16 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v8 = (__int64 *)(v31 + 8LL * v29);
        v15 = *v8;
        v14 = v32 + 1;
        if ( v6 != (__int64 *)*v8 )
        {
          while ( v15 != -8 )
          {
            if ( v15 == -16 && !v28 )
              v28 = v8;
            v29 = v16 & (v27 + v29);
            v8 = (__int64 *)(v31 + 8LL * v29);
            v15 = *v8;
            if ( v6 == (__int64 *)*v8 )
              goto LABEL_19;
            ++v27;
          }
          if ( v28 )
            v8 = v28;
        }
        goto LABEL_19;
      }
LABEL_72:
      LODWORD(v32) = v32 + 1;
      BUG();
    }
LABEL_48:
    sub_13B3B90((__int64)&v30, 2 * v33);
    if ( !(_DWORD)v33 )
      goto LABEL_72;
    v15 = (unsigned int)(v33 - 1);
    v16 = (unsigned int)v15 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v8 = (__int64 *)(v31 + 8 * v16);
    v5 = *v8;
    v14 = v32 + 1;
    if ( v6 != (__int64 *)*v8 )
    {
      v25 = 1;
      v26 = 0;
      while ( v5 != -8 )
      {
        if ( v5 == -16 && !v26 )
          v26 = v8;
        v16 = (unsigned int)v15 & (v25 + (_DWORD)v16);
        v8 = (__int64 *)(v31 + 8LL * (unsigned int)v16);
        v5 = *v8;
        if ( v6 == (__int64 *)*v8 )
          goto LABEL_19;
        ++v25;
      }
      if ( v26 )
        v8 = v26;
    }
LABEL_19:
    LODWORD(v32) = v14;
    if ( *v8 != -8 )
      --HIDWORD(v32);
    *v8 = (__int64)v6;
    v17 = *((_BYTE *)v6 + 16);
    if ( v17 > 0x17u )
      break;
    if ( v17 > 0x10u )
      goto LABEL_23;
    v24 = v6 == (__int64 *)sub_15A06D0((__int64 **)*v6, v5, v16, v15);
    LODWORD(i) = v35;
    if ( !v24 )
      v1 = 0;
LABEL_27:
    v2 = v31;
  }
  if ( (unsigned int)v17 - 60 <= 0xC )
  {
    v18 = sub_1649C60(v6);
    i = (unsigned int)v35;
    if ( (unsigned int)v35 >= HIDWORD(v35) )
      goto LABEL_34;
    goto LABEL_26;
  }
  switch ( v17 )
  {
    case '8':
      v18 = v6[-3 * (*((_DWORD *)v6 + 5) & 0xFFFFFFF)];
      i = (unsigned int)v35;
      if ( (unsigned int)v35 >= HIDWORD(v35) )
        goto LABEL_34;
      goto LABEL_26;
    case 'M':
      v21 = 24LL * (*((_DWORD *)v6 + 5) & 0xFFFFFFF);
      if ( (*((_BYTE *)v6 + 23) & 0x40) != 0 )
      {
        v22 = (__int64 *)*(v6 - 1);
        v6 = &v22[(unsigned __int64)v21 / 8];
      }
      else
      {
        v22 = &v6[v21 / 0xFFFFFFFFFFFFFFF8LL];
      }
      for ( i = (unsigned int)v35; v6 != v22; LODWORD(v35) = v35 + 1 )
      {
        v23 = *v22;
        if ( HIDWORD(v35) <= (unsigned int)i )
        {
          sub_16CD150(&v34, v36, 0, 8);
          i = (unsigned int)v35;
        }
        v22 += 3;
        v34[i] = v23;
        i = (unsigned int)(v35 + 1);
      }
      goto LABEL_27;
    case 'O':
      v19 = *(v6 - 6);
      v20 = (unsigned int)v35;
      if ( (unsigned int)v35 >= HIDWORD(v35) )
      {
        sub_16CD150(&v34, v36, 0, 8);
        v20 = (unsigned int)v35;
      }
      v34[v20] = v19;
      i = (unsigned int)(v35 + 1);
      LODWORD(v35) = i;
      v18 = *(v6 - 3);
      if ( HIDWORD(v35) <= (unsigned int)i )
      {
LABEL_34:
        sub_16CD150(&v34, v36, 0, 8);
        i = (unsigned int)v35;
      }
LABEL_26:
      v34[i] = v18;
      LODWORD(i) = v35 + 1;
      LODWORD(v35) = v35 + 1;
      goto LABEL_27;
  }
LABEL_23:
  v2 = v31;
  v12 = 1;
LABEL_7:
  j___libc_free_0(v2);
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
  return v12;
}
