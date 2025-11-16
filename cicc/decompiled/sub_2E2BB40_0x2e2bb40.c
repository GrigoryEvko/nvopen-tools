// Function: sub_2E2BB40
// Address: 0x2e2bb40
//
__int64 __fastcall sub_2E2BB40(__m128i *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // r12
  __int64 v12; // r14
  __int64 v13; // r15
  char v14; // di
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 *v17; // r13
  __int64 *v18; // r14
  __int64 *v19; // r8
  __int64 v20; // rsi
  __int64 *v21; // rax
  unsigned int v22; // ecx
  unsigned int v23; // edi
  unsigned int v24; // edx
  unsigned int v25; // r15d
  __int64 *v26; // rax
  __int64 *v27; // rdx
  __int64 v29; // [rsp+0h] [rbp-90h] BYREF
  __int64 *v30; // [rsp+8h] [rbp-88h]
  __int64 v31; // [rsp+10h] [rbp-80h]
  int v32; // [rsp+18h] [rbp-78h]
  unsigned __int8 v33; // [rsp+1Ch] [rbp-74h]
  char v34; // [rsp+20h] [rbp-70h] BYREF

  v31 = 8;
  v11 = (__int64 *)sub_2E29D60(a1, a2, a3, a4, a5, a6);
  v32 = 0;
  v12 = v11[4];
  v13 = v11[5];
  v33 = 1;
  v29 = 0;
  v30 = (__int64 *)&v34;
  if ( v12 == v13 )
  {
    v17 = *(__int64 **)(a3 + 112);
    v18 = &v17[*(unsigned int *)(a3 + 120)];
    if ( v17 == v18 )
      return 0;
    while ( 1 )
    {
LABEL_10:
      v19 = (__int64 *)*v11;
      v20 = *v17;
      if ( v11 == (__int64 *)*v11 )
        goto LABEL_19;
      v21 = (__int64 *)v11[3];
      v22 = *(_DWORD *)(v20 + 24);
      if ( v11 == v21 )
      {
        v21 = (__int64 *)v11[1];
        v24 = v22 >> 7;
        v11[3] = (__int64)v21;
        v23 = *((_DWORD *)v21 + 4);
        if ( v22 >> 7 == v23 )
        {
          if ( v11 != v21 )
          {
LABEL_36:
            if ( (v21[((v22 >> 6) & 1) + 3] & (1LL << v22)) != 0 )
            {
LABEL_37:
              v14 = v33;
              v25 = 1;
              goto LABEL_38;
            }
            goto LABEL_19;
          }
          goto LABEL_19;
        }
      }
      else
      {
        v23 = *((_DWORD *)v21 + 4);
        v24 = v22 >> 7;
        if ( v22 >> 7 == v23 )
          goto LABEL_36;
      }
      if ( v24 >= v23 )
      {
        if ( v11 == v21 )
        {
LABEL_47:
          v11[3] = (__int64)v21;
          goto LABEL_19;
        }
        while ( v24 > v23 )
        {
          v21 = (__int64 *)*v21;
          if ( v11 == v21 )
            goto LABEL_47;
          v23 = *((_DWORD *)v21 + 4);
        }
LABEL_17:
        v11[3] = (__int64)v21;
        if ( v11 == v21 )
          goto LABEL_19;
      }
      else
      {
        if ( v21 != v19 )
        {
          while ( 1 )
          {
            v21 = (__int64 *)v21[1];
            if ( v19 == v21 )
              break;
            if ( v24 >= *((_DWORD *)v21 + 4) )
              goto LABEL_17;
          }
        }
        v11[3] = (__int64)v21;
      }
      if ( v24 == *((_DWORD *)v21 + 4) )
        goto LABEL_36;
LABEL_19:
      v25 = v33;
      if ( v33 )
      {
        v26 = v30;
        v27 = &v30[HIDWORD(v31)];
        if ( v30 != v27 )
        {
          while ( v20 != *v26 )
          {
            if ( v27 == ++v26 )
              goto LABEL_26;
          }
          return v25;
        }
      }
      else if ( sub_C8CA60((__int64)&v29, v20) )
      {
        goto LABEL_37;
      }
LABEL_26:
      if ( v18 == ++v17 )
      {
        v25 = 0;
        if ( v33 )
          return v25;
LABEL_28:
        _libc_free((unsigned __int64)v30);
        return v25;
      }
    }
  }
  v14 = 1;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v15 = *(_QWORD *)(*(_QWORD *)v12 + 24LL);
        if ( v14 )
          break;
LABEL_29:
        v12 += 8;
        sub_C8CC70((__int64)&v29, v15, (__int64)v7, v8, v9, v10);
        v14 = v33;
        if ( v13 == v12 )
          goto LABEL_9;
      }
      v16 = v30;
      v8 = HIDWORD(v31);
      v7 = &v30[HIDWORD(v31)];
      if ( v30 != v7 )
        break;
LABEL_31:
      if ( HIDWORD(v31) >= (unsigned int)v31 )
        goto LABEL_29;
      v8 = (unsigned int)(HIDWORD(v31) + 1);
      v12 += 8;
      ++HIDWORD(v31);
      *v7 = v15;
      v14 = v33;
      ++v29;
      if ( v13 == v12 )
        goto LABEL_9;
    }
    while ( v15 != *v16 )
    {
      if ( v7 == ++v16 )
        goto LABEL_31;
    }
    v12 += 8;
  }
  while ( v13 != v12 );
LABEL_9:
  v17 = *(__int64 **)(a3 + 112);
  v18 = &v17[*(unsigned int *)(a3 + 120)];
  if ( v17 != v18 )
    goto LABEL_10;
  v25 = 0;
LABEL_38:
  if ( !v14 )
    goto LABEL_28;
  return v25;
}
