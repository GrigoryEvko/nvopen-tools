// Function: sub_3891B00
// Address: 0x3891b00
//
__int64 __fastcall sub_3891B00(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  unsigned int v6; // eax
  int v7; // eax
  int v8; // eax
  _BOOL4 v9; // r15d
  char v11; // al
  char v12; // al
  size_t v13; // r15
  __int64 v14; // r14
  unsigned int v15; // r9d
  __int64 **v16; // r10
  __int64 *v17; // rcx
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  unsigned int v20; // r15d
  unsigned __int64 v21; // r14
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  char v27; // di
  __int64 v28; // rax
  unsigned int v29; // r9d
  __int64 **v30; // r10
  _QWORD *v31; // rcx
  _BYTE *v32; // rdi
  __int64 **v33; // rax
  __int64 **v34; // rax
  __int64 v35; // rax
  _BYTE *v36; // rax
  const char *v37; // rax
  unsigned __int64 v38; // rsi
  __int64 **v39; // [rsp+0h] [rbp-80h]
  __int64 **v40; // [rsp+0h] [rbp-80h]
  unsigned int v41; // [rsp+8h] [rbp-78h]
  __int64 **v42; // [rsp+8h] [rbp-78h]
  unsigned int v43; // [rsp+8h] [rbp-78h]
  unsigned int v44; // [rsp+10h] [rbp-70h]
  _QWORD *v45; // [rsp+10h] [rbp-70h]
  _QWORD *v46; // [rsp+18h] [rbp-68h]
  _QWORD *v47; // [rsp+18h] [rbp-68h]
  unsigned __int8 *src; // [rsp+20h] [rbp-60h]
  void *srcb; // [rsp+20h] [rbp-60h]
  void *srcc; // [rsp+20h] [rbp-60h]
  __int64 *srca; // [rsp+20h] [rbp-60h]
  unsigned __int64 v52; // [rsp+28h] [rbp-58h]
  int v53[4]; // [rsp+30h] [rbp-50h] BYREF
  char v54; // [rsp+40h] [rbp-40h]
  char v55; // [rsp+41h] [rbp-3Fh]

  v52 = *(_QWORD *)(a1 + 56);
  v6 = *(_DWORD *)(a1 + 64);
  if ( v6 == 369 )
  {
    v19 = *(_QWORD *)(a1 + 776);
    v20 = *(_DWORD *)(a1 + 104);
    v21 = a1 + 768;
    if ( !v19 )
      goto LABEL_48;
    do
    {
      while ( 1 )
      {
        v22 = *(_QWORD *)(v19 + 16);
        v23 = *(_QWORD *)(v19 + 24);
        if ( v20 <= *(_DWORD *)(v19 + 32) )
          break;
        v19 = *(_QWORD *)(v19 + 24);
        if ( !v23 )
          goto LABEL_46;
      }
      v21 = v19;
      v19 = *(_QWORD *)(v19 + 16);
    }
    while ( v22 );
LABEL_46:
    if ( a1 + 768 == v21 || v20 < *(_DWORD *)(v21 + 32) )
    {
LABEL_48:
      srcb = (void *)v21;
      v46 = (_QWORD *)(a1 + 768);
      v24 = sub_22077B0(0x38u);
      *(_DWORD *)(v24 + 32) = v20;
      v21 = v24;
      *(_QWORD *)(v24 + 40) = 0;
      *(_QWORD *)(v24 + 48) = 0;
      v25 = sub_3891A00((_QWORD *)(a1 + 760), (__int64)srcb, (unsigned int *)(v24 + 32));
      if ( v26 )
      {
        v27 = v25 || v46 == (_QWORD *)v26 || v20 < *(_DWORD *)(v26 + 32);
        sub_220F040(v27, v21, (_QWORD *)v26, v46);
        ++*(_QWORD *)(a1 + 800);
      }
      else
      {
        srcc = (void *)v25;
        j_j___libc_free_0(v21);
        v21 = (unsigned __int64)srcc;
      }
    }
    v18 = *(_QWORD *)(v21 + 40);
    if ( !v18 )
    {
      v18 = sub_16440F0(*(_QWORD *)a1);
      *(_QWORD *)(v21 + 40) = v18;
      *(_QWORD *)(v21 + 48) = *(_QWORD *)(a1 + 56);
    }
    goto LABEL_39;
  }
  if ( v6 > 0x171 )
  {
    if ( v6 != 375 )
    {
      if ( v6 == 388 )
      {
        *a2 = *(_QWORD *)(a1 + 112);
        v8 = sub_3887100(a1 + 8);
        *(_DWORD *)(a1 + 64) = v8;
        goto LABEL_8;
      }
      return sub_38814C0(a1 + 8, v52, a3);
    }
    v13 = *(_QWORD *)(a1 + 80);
    v14 = a1 + 728;
    src = *(unsigned __int8 **)(a1 + 72);
    v15 = sub_16D19C0(a1 + 728, src, v13);
    v16 = (__int64 **)(*(_QWORD *)(a1 + 728) + 8LL * v15);
    v17 = *v16;
    if ( *v16 )
    {
      if ( v17 != (__int64 *)-8LL )
      {
LABEL_37:
        v18 = v17[1];
        if ( !v18 )
        {
          srca = v17;
          v18 = sub_1644060(*(_QWORD *)a1, *(const void **)(a1 + 72), *(_QWORD *)(a1 + 80));
          srca[1] = v18;
          srca[2] = *(_QWORD *)(a1 + 56);
        }
LABEL_39:
        *a2 = v18;
        v8 = sub_3887100(a1 + 8);
        for ( *(_DWORD *)(a1 + 64) = v8; ; *(_DWORD *)(a1 + 64) = v8 )
        {
          while ( 1 )
          {
LABEL_8:
            while ( v8 == 12 )
            {
              if ( (unsigned __int8)sub_38935A0(a1, a2) )
                return 1;
              v8 = *(_DWORD *)(a1 + 64);
            }
            if ( v8 != 89 )
              break;
            v12 = *(_BYTE *)(*a2 + 8);
            if ( v12 == 7 )
            {
              v55 = 1;
              v37 = "basic block pointers are invalid";
              goto LABEL_73;
            }
            if ( !v12 )
            {
              v55 = 1;
              v37 = "pointers to void are invalid; use i8* instead";
              goto LABEL_73;
            }
            v9 = sub_1643F60(*a2);
            if ( !v9 )
            {
              v55 = 1;
              v37 = "pointer to this type is invalid";
              goto LABEL_73;
            }
            if ( (unsigned __int8)sub_388BF60(a1, v53)
              || (unsigned __int8)sub_388AF10(a1, 5, "expected '*' in address space") )
            {
              return v9;
            }
            *a2 = sub_1646BA0((__int64 *)*a2, v53[0]);
            v8 = *(_DWORD *)(a1 + 64);
          }
          if ( v8 != 5 )
            break;
          v11 = *(_BYTE *)(*a2 + 8);
          if ( v11 == 7 )
          {
            v55 = 1;
            v37 = "basic block pointers are invalid";
LABEL_73:
            v38 = *(_QWORD *)(a1 + 56);
            *(_QWORD *)v53 = v37;
            v54 = 3;
            return (_BOOL4)sub_38814C0(a1 + 8, v38, (__int64)v53);
          }
          if ( !v11 )
          {
            v55 = 1;
            v37 = "pointers to void are invalid - use i8* instead";
            goto LABEL_73;
          }
          if ( !sub_1643F60(*a2) )
          {
            v55 = 1;
            v37 = "pointer to this type is invalid";
            goto LABEL_73;
          }
          *a2 = sub_1646BA0((__int64 *)*a2, 0);
          v8 = sub_3887100(a1 + 8);
        }
        v9 = 0;
        if ( !a4 && !*(_BYTE *)(*a2 + 8) )
        {
          v55 = 1;
          v54 = 3;
          *(_QWORD *)v53 = "void type only allowed for function results";
          return (_BOOL4)sub_38814C0(a1 + 8, v52, (__int64)v53);
        }
        return v9;
      }
      --*(_DWORD *)(a1 + 744);
    }
    v39 = v16;
    v41 = v15;
    v28 = malloc(v13 + 25);
    v29 = v41;
    v30 = v39;
    v31 = (_QWORD *)v28;
    if ( !v28 )
    {
      if ( v13 == -25 )
      {
        v35 = malloc(1u);
        v31 = 0;
        v29 = v41;
        v30 = v39;
        if ( v35 )
        {
          v32 = (_BYTE *)(v35 + 24);
          v31 = (_QWORD *)v35;
          goto LABEL_69;
        }
      }
      v40 = v30;
      v43 = v29;
      v45 = v31;
      sub_16BD1C0("Allocation failed", 1u);
      v31 = v45;
      v29 = v43;
      v30 = v40;
    }
    v32 = v31 + 3;
    if ( v13 + 1 <= 1 )
    {
LABEL_58:
      v32[v13] = 0;
      *v31 = v13;
      v31[1] = 0;
      v31[2] = 0;
      *v30 = v31;
      ++*(_DWORD *)(a1 + 740);
      v33 = (__int64 **)(*(_QWORD *)(a1 + 728) + 8LL * (unsigned int)sub_16D1CD0(v14, v29));
      v17 = *v33;
      if ( !*v33 || v17 == (__int64 *)-8LL )
      {
        v34 = v33 + 1;
        do
        {
          do
            v17 = *v34++;
          while ( !v17 );
        }
        while ( v17 == (__int64 *)-8LL );
      }
      goto LABEL_37;
    }
LABEL_69:
    v42 = v30;
    v44 = v29;
    v47 = v31;
    v36 = memcpy(v32, src, v13);
    v30 = v42;
    v29 = v44;
    v31 = v47;
    v32 = v36;
    goto LABEL_58;
  }
  switch ( v6 )
  {
    case 8u:
      if ( !(unsigned __int8)sub_3892330(a1, a2, 0) )
        goto LABEL_7;
      return 1;
    case 0xAu:
      v7 = sub_3887100(a1 + 8);
      *(_DWORD *)(a1 + 64) = v7;
      if ( v7 == 8 )
      {
        if ( !(unsigned __int8)sub_3892330(a1, a2, 1)
          && !(unsigned __int8)sub_388AF10(a1, 11, "expected '>' at end of packed struct") )
        {
          goto LABEL_7;
        }
      }
      else if ( !(unsigned __int8)sub_38923B0(a1, a2, 1) )
      {
LABEL_7:
        v8 = *(_DWORD *)(a1 + 64);
        goto LABEL_8;
      }
      return 1;
    case 6u:
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      if ( !(unsigned __int8)sub_38923B0(a1, a2, 0) )
        goto LABEL_7;
      return 1;
  }
  return sub_38814C0(a1 + 8, v52, a3);
}
