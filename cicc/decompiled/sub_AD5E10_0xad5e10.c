// Function: sub_AD5E10
// Address: 0xad5e10
//
__int64 __fastcall sub_AD5E10(__int64 n, unsigned __int8 *a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rdx
  char v4; // al
  __int64 *v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rax
  void *v8; // rdi
  __int64 v9; // r12
  __int64 *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 **v14; // r14
  _DWORD *v15; // rax
  __int64 v16; // r15
  __int64 v17; // r14
  unsigned __int8 *v18; // rax
  __int64 v19; // r12
  unsigned int v20; // eax
  _DWORD *v21; // rdx
  __int64 v22; // rax
  __int64 *v23; // rax
  char *v24; // rax
  char *v25; // rdx
  __int64 v26; // rdx
  unsigned int v27; // eax
  unsigned int v28; // eax
  unsigned int v29; // edx
  __int64 v30; // rcx
  void *s; // [rsp+10h] [rbp-140h] BYREF
  __int64 v32; // [rsp+18h] [rbp-138h]
  _DWORD v33[76]; // [rsp+20h] [rbp-130h] BYREF

  v2 = n;
  if ( !BYTE4(n) )
  {
    if ( sub_AC30F0((__int64)a2) )
    {
LABEL_3:
      v4 = *a2;
LABEL_4:
      if ( (unsigned __int8)(v4 - 17) <= 1u && sub_AC5240(*((_QWORD *)a2 + 1)) )
        return sub_AD9660((unsigned int)n);
      s = v33;
      v32 = 0x2000000000LL;
      if ( (unsigned int)n > 0x20 )
      {
        sub_C8D5F0(&s, v33, (unsigned int)n, 8);
        v24 = (char *)s;
        v25 = (char *)s + 8 * (unsigned int)n;
        do
        {
          *(_QWORD *)v24 = a2;
          v24 += 8;
        }
        while ( v25 != v24 );
        LODWORD(v32) = n;
        v5 = (__int64 *)s;
      }
      else
      {
        v5 = (__int64 *)v33;
        if ( v2 )
        {
          v15 = v33;
          do
          {
            *(_QWORD *)v15 = a2;
            v15 += 2;
          }
          while ( &v33[2 * v2] != v15 );
          v5 = (__int64 *)s;
        }
        LODWORD(v32) = v2;
      }
      v6 = v2;
      v7 = sub_AD3730(v5, v2);
      v8 = s;
      v9 = v7;
      if ( s != v33 )
        goto LABEL_9;
      return v9;
    }
    if ( !(_BYTE)qword_4F81308 || *a2 != 17 )
    {
      if ( !(_BYTE)qword_4F81228 )
        goto LABEL_3;
      v4 = *a2;
      if ( *a2 != 18 )
        goto LABEL_4;
LABEL_15:
      v11 = (__int64 *)sub_BD5C60(a2, a2, v3);
      return sub_ACE4D0(v11, n, (_QWORD *)a2 + 3, v12, v13);
    }
    goto LABEL_40;
  }
  if ( !sub_AC30F0((__int64)a2) )
  {
    if ( (_BYTE)qword_4F81148 && *a2 == 17 )
    {
LABEL_40:
      v23 = (__int64 *)sub_BD5C60(a2, a2, v3);
      return sub_ACD980(v23, n, (__int64)(a2 + 24));
    }
    if ( (_BYTE)qword_4F81068 && *a2 == 18 )
      goto LABEL_15;
  }
  v14 = (__int64 **)sub_BCE1B0(*((_QWORD *)a2 + 1), n);
  if ( !sub_AC30F0((__int64)a2) )
  {
    if ( (unsigned int)*a2 - 12 <= 1 )
      return sub_ACA8A0(v14);
    v16 = sub_BCB2E0(*v14);
    v17 = sub_ACADE0(v14);
    v18 = (unsigned __int8 *)sub_AD64C0(v16, 0, 0);
    s = v33;
    v19 = sub_AD5A90(v17, a2, v18, 0);
    v32 = 0x800000000LL;
    if ( (unsigned int)n > 8 )
    {
      sub_C8D5F0(&s, v33, (unsigned int)n, 4);
      memset(s, 0, 4LL * (unsigned int)n);
      LODWORD(v32) = n;
      v21 = s;
    }
    else
    {
      if ( (_DWORD)n )
      {
        v20 = 4 * n;
        if ( 4LL * (unsigned int)n )
        {
          if ( v20 >= 8 )
          {
            v26 = v20;
            v27 = v20 - 1;
            *(_QWORD *)((char *)&v33[-2] + v26) = 0;
            if ( v27 >= 8 )
            {
              v28 = v27 & 0xFFFFFFF8;
              v29 = 0;
              do
              {
                v30 = v29;
                v29 += 8;
                *(_QWORD *)((char *)v33 + v30) = 0;
              }
              while ( v29 < v28 );
            }
          }
          else if ( (v20 & 4) != 0 )
          {
            v33[0] = 0;
            v33[v20 / 4 - 1] = 0;
          }
          else if ( v20 )
          {
            LOBYTE(v33[0]) = 0;
          }
        }
      }
      LODWORD(v32) = n;
      v21 = v33;
    }
    v6 = v17;
    v22 = sub_AD5CE0(v19, v17, v21, (unsigned int)n, 0);
    v8 = s;
    v9 = v22;
    if ( s != v33 )
LABEL_9:
      _libc_free(v8, v6);
    return v9;
  }
  return sub_AC9350(v14);
}
