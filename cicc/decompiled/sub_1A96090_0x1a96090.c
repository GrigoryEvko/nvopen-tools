// Function: sub_1A96090
// Address: 0x1a96090
//
void __fastcall sub_1A96090(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v3; // rbx
  __int64 *v4; // r15
  __int64 *v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 *v8; // rdx
  __int64 *v9; // rbx
  __int64 *v10; // rax
  char v11; // dl
  char v12; // r8
  __int64 *v13; // rdx
  __int64 v14; // rsi
  __int64 *v15; // rax
  char v16; // dl
  char v17; // r8
  __int64 *v18; // rdx
  __int64 v19; // rsi
  __int64 *v20; // rax
  char v21; // dl
  char v22; // r8
  __int64 *v23; // rdx
  __int64 v24; // rsi
  char v25; // dl
  char v26; // r8
  __int64 v27; // rsi
  __int64 *v28; // rdi
  __int64 *v29; // rcx
  __int64 *v30; // r14
  char v31; // dl
  __int64 v32; // rsi
  __int64 *v33; // rax
  __int64 *v34; // rdi
  __int64 *v35; // rcx
  __int64 *v36; // rdi
  __int64 *v37; // rcx
  __int64 *v38; // rdi
  __int64 *v39; // rcx
  __int64 *v40; // rdi
  __int64 *v41; // rcx
  __int64 *v42; // rax
  __int64 v43; // rbx
  unsigned __int64 v44; // rdi
  __int64 v45; // [rsp+0h] [rbp-A0h] BYREF
  __int64 *v46; // [rsp+8h] [rbp-98h]
  __int64 *v47; // [rsp+10h] [rbp-90h]
  __int64 v48; // [rsp+18h] [rbp-88h]
  int v49; // [rsp+20h] [rbp-80h]
  _BYTE v50[120]; // [rsp+28h] [rbp-78h] BYREF

  v1 = (__int64 *)v50;
  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(__int64 **)a1;
  v45 = 0;
  v46 = (__int64 *)v50;
  v3 *= 8;
  v47 = (__int64 *)v50;
  v5 = (__int64 *)((char *)v4 + v3);
  v48 = 8;
  v6 = v3 >> 3;
  v7 = v3 >> 5;
  v49 = 0;
  if ( !v7 )
  {
LABEL_63:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
        {
LABEL_66:
          v4 = v5;
          goto LABEL_67;
        }
LABEL_86:
        if ( (unsigned __int8)sub_1A96000((__int64)&v45, *v4) )
          goto LABEL_18;
        goto LABEL_66;
      }
      if ( (unsigned __int8)sub_1A96000((__int64)&v45, *v4) )
        goto LABEL_18;
      ++v4;
    }
    if ( (unsigned __int8)sub_1A96000((__int64)&v45, *v4) )
      goto LABEL_18;
    ++v4;
    goto LABEL_86;
  }
  v8 = (__int64 *)v50;
  v9 = &v4[4 * v7];
  while ( 1 )
  {
    while ( 1 )
    {
      v27 = *v4;
      if ( v1 != v8 )
        goto LABEL_3;
      v28 = &v1[HIDWORD(v48)];
      if ( v28 != v1 )
      {
        v29 = 0;
        do
        {
          if ( v27 == *v1 )
            goto LABEL_18;
          if ( *v1 == -2 )
            v29 = v1;
          ++v1;
        }
        while ( v28 != v1 );
        if ( v29 )
        {
          *v29 = v27;
          v13 = v47;
          --v49;
          v10 = v46;
          ++v45;
          goto LABEL_4;
        }
      }
      if ( HIDWORD(v48) < (unsigned int)v48 )
      {
        ++HIDWORD(v48);
        *v28 = v27;
        v10 = v46;
        ++v45;
        v13 = v47;
      }
      else
      {
LABEL_3:
        sub_16CCBA0((__int64)&v45, v27);
        v10 = v46;
        v12 = v11;
        v13 = v47;
        if ( !v12 )
          goto LABEL_18;
      }
LABEL_4:
      v14 = v4[1];
      if ( v10 != v13 )
        goto LABEL_5;
      v36 = &v10[HIDWORD(v48)];
      if ( v36 == v10 )
        goto LABEL_90;
      v37 = 0;
      do
      {
        if ( v14 == *v10 )
          goto LABEL_39;
        if ( *v10 == -2 )
          v37 = v10;
        ++v10;
      }
      while ( v36 != v10 );
      if ( !v37 )
      {
LABEL_90:
        if ( HIDWORD(v48) >= (unsigned int)v48 )
        {
LABEL_5:
          sub_16CCBA0((__int64)&v45, v14);
          v15 = v46;
          v17 = v16;
          v18 = v47;
          if ( !v17 )
          {
LABEL_39:
            ++v4;
            goto LABEL_18;
          }
          goto LABEL_6;
        }
        ++HIDWORD(v48);
        *v36 = v14;
        v15 = v46;
        ++v45;
        v18 = v47;
      }
      else
      {
        *v37 = v14;
        v18 = v47;
        --v49;
        v15 = v46;
        ++v45;
      }
LABEL_6:
      v19 = v4[2];
      if ( v18 != v15 )
        goto LABEL_7;
      v38 = &v15[HIDWORD(v48)];
      if ( v15 != v38 )
      {
        v39 = 0;
        do
        {
          if ( v19 == *v15 )
            goto LABEL_46;
          if ( *v15 == -2 )
            v39 = v15;
          ++v15;
        }
        while ( v38 != v15 );
        if ( v39 )
        {
          *v39 = v19;
          v23 = v47;
          --v49;
          v20 = v46;
          ++v45;
          goto LABEL_8;
        }
      }
      if ( HIDWORD(v48) < (unsigned int)v48 )
      {
        ++HIDWORD(v48);
        *v38 = v19;
        v20 = v46;
        ++v45;
        v23 = v47;
      }
      else
      {
LABEL_7:
        sub_16CCBA0((__int64)&v45, v19);
        v20 = v46;
        v22 = v21;
        v23 = v47;
        if ( !v22 )
        {
LABEL_46:
          v4 += 2;
          goto LABEL_18;
        }
      }
LABEL_8:
      v24 = v4[3];
      if ( v20 != v23 )
        goto LABEL_9;
      v40 = &v20[HIDWORD(v48)];
      if ( v20 == v40 )
        break;
      v41 = 0;
      do
      {
        if ( v24 == *v20 )
          goto LABEL_53;
        if ( *v20 == -2 )
          v41 = v20;
        ++v20;
      }
      while ( v40 != v20 );
      if ( !v41 )
        break;
      v4 += 4;
      *v41 = v24;
      v8 = v47;
      --v49;
      v1 = v46;
      ++v45;
      if ( v9 == v4 )
      {
LABEL_62:
        v6 = v5 - v4;
        goto LABEL_63;
      }
    }
    if ( HIDWORD(v48) < (unsigned int)v48 )
    {
      ++HIDWORD(v48);
      *v40 = v24;
      v1 = v46;
      ++v45;
      v8 = v47;
      goto LABEL_10;
    }
LABEL_9:
    sub_16CCBA0((__int64)&v45, v24);
    v1 = v46;
    v26 = v25;
    v8 = v47;
    if ( !v26 )
      break;
LABEL_10:
    v4 += 4;
    if ( v9 == v4 )
      goto LABEL_62;
  }
LABEL_53:
  v4 += 3;
LABEL_18:
  if ( v5 != v4 )
  {
    v30 = v4 + 1;
    if ( v5 != v4 + 1 )
    {
      while ( 2 )
      {
        v32 = *v30;
        v33 = v46;
        if ( v47 != v46 )
          goto LABEL_21;
        v34 = &v46[HIDWORD(v48)];
        if ( v46 != v34 )
        {
          v35 = 0;
          while ( v32 != *v33 )
          {
            if ( *v33 == -2 )
              v35 = v33;
            if ( v34 == ++v33 )
            {
              if ( !v35 )
                goto LABEL_80;
              *v35 = v32;
              --v49;
              ++v45;
              goto LABEL_22;
            }
          }
LABEL_23:
          if ( v5 == ++v30 )
            goto LABEL_67;
          continue;
        }
        break;
      }
LABEL_80:
      if ( HIDWORD(v48) < (unsigned int)v48 )
      {
        ++HIDWORD(v48);
        *v34 = v32;
        ++v45;
      }
      else
      {
LABEL_21:
        sub_16CCBA0((__int64)&v45, v32);
        if ( !v31 )
          goto LABEL_23;
      }
LABEL_22:
      *v4++ = *v30;
      goto LABEL_23;
    }
  }
LABEL_67:
  v42 = *(__int64 **)a1;
  v43 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v5;
  if ( v5 != (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
  {
    memmove(v4, v5, *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v5);
    v42 = *(__int64 **)a1;
  }
  v44 = (unsigned __int64)v47;
  *(_DWORD *)(a1 + 8) = ((char *)v4 + v43 - (char *)v42) >> 3;
  if ( (__int64 *)v44 != v46 )
    _libc_free(v44);
}
