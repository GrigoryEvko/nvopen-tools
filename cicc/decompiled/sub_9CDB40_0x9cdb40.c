// Function: sub_9CDB40
// Address: 0x9cdb40
//
unsigned __int64 *__fastcall sub_9CDB40(unsigned __int64 *a1, unsigned __int64 *a2, unsigned __int64 *a3)
{
  unsigned __int64 v6; // rdi
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 *v10; // rax
  unsigned __int64 v11; // rbx
  char *v12; // rsi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rax
  __int64 *v15; // rax
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // r14
  unsigned __int64 *v18; // rbx
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rdx
  char *v21; // rbx
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdi
  __int64 *v27; // r14
  char *v28; // rsi
  __int64 v29; // rdx
  unsigned __int64 v30; // r8
  __int64 *v31; // rcx
  _QWORD *v32; // r8
  _QWORD *v33; // r15
  _QWORD *i; // r12
  unsigned __int64 v35; // rdi
  char *v36; // rsi
  __int64 v37; // rdi
  _QWORD *v38; // [rsp+0h] [rbp-50h]
  __int64 *v39; // [rsp+8h] [rbp-48h]
  unsigned __int64 v40; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v41[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( (*a2 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    *a2 = 0;
LABEL_3:
    *a1 = 0;
    *a1 = *a3 | 1;
    *a3 = 0;
    return a1;
  }
  *a2 = *a2 & 0xFFFFFFFFFFFFFFFELL | 1;
  v6 = *a3 & 0xFFFFFFFFFFFFFFFELL;
  if ( !v6 )
  {
    *a3 = 0;
LABEL_6:
    *a1 = 0;
    *a1 = *a2 | 1;
    *a2 = 0;
    return a1;
  }
  *a3 = v6 | 1;
  if ( (*a2 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_64;
  if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)(*a2 & 0xFFFFFFFFFFFFFFFELL) + 48LL))(
         *a2 & 0xFFFFFFFFFFFFFFFELL,
         &unk_4F84052) )
  {
    v27 = (__int64 *)(*a2 & 0xFFFFFFFFFFFFFFFELL);
    if ( (*a3 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v28 = (char *)&unk_4F84052;
      if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)(*a3 & 0xFFFFFFFFFFFFFFFELL) + 48LL))(
             *a3 & 0xFFFFFFFFFFFFFFFELL,
             &unk_4F84052) )
      {
        v30 = *a3;
        *a3 = 0;
        v31 = v27 + 1;
        v32 = (_QWORD *)(v30 & 0xFFFFFFFFFFFFFFFELL);
        v33 = (_QWORD *)v32[2];
        for ( i = (_QWORD *)v32[1]; v33 != i; ++i )
        {
          v28 = (char *)v27[2];
          if ( v28 == (char *)v27[3] )
          {
            v38 = v32;
            v39 = v31;
            sub_9CD990(v31, v28, i);
            v32 = v38;
            v31 = v39;
          }
          else
          {
            if ( v28 )
            {
              *(_QWORD *)v28 = *i;
              *i = 0;
              v28 = (char *)v27[2];
            }
            v28 += 8;
            v27[2] = (__int64)v28;
          }
        }
        (*(void (__fastcall **)(_QWORD *, char *, __int64, __int64 *))(*v32 + 8LL))(v32, v28, v29, v31);
        goto LABEL_6;
      }
      v35 = *a3 & 0xFFFFFFFFFFFFFFFELL;
    }
    else
    {
      v35 = 0;
    }
    v41[0] = v35;
    *a3 = 0;
    v36 = (char *)v27[2];
    if ( v36 == (char *)v27[3] )
    {
      sub_9CD990(v27 + 1, v36, v41);
      v35 = v41[0];
    }
    else
    {
      if ( v36 )
      {
        *(_QWORD *)v36 = v35;
        v27[2] += 8;
        goto LABEL_6;
      }
      v27[2] = 8;
    }
    if ( v35 )
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v35 + 8LL))(v35);
    goto LABEL_6;
  }
  v6 = *a3 & 0xFFFFFFFFFFFFFFFELL;
  if ( v6 )
  {
LABEL_64:
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v6 + 48LL))(v6, &unk_4F84052) )
    {
      v13 = *a2;
      v14 = *a3;
      *a2 = 0;
      v15 = (__int64 *)(v14 & 0xFFFFFFFFFFFFFFFELL);
      v16 = v13 & 0xFFFFFFFFFFFFFFFELL;
      v41[0] = v16;
      v17 = (unsigned __int64 *)v15[1];
      v18 = (unsigned __int64 *)v15[2];
      if ( v18 == (unsigned __int64 *)v15[3] )
      {
        sub_9CD990(v15 + 1, (char *)v15[1], v41);
      }
      else
      {
        if ( v17 == v18 )
        {
          if ( v17 )
          {
            *v17 = v16;
            v15[2] += 8;
            goto LABEL_3;
          }
          v15[2] = 8;
LABEL_31:
          if ( v16 )
            (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v16 + 8LL))(v16);
          goto LABEL_3;
        }
        if ( v18 )
        {
          v19 = *(v18 - 1);
          *(v18 - 1) = 0;
          *v18 = v19;
          v18 = (unsigned __int64 *)v15[2];
        }
        v20 = v18 + 1;
        v21 = (char *)(v18 - 1);
        v15[2] = (__int64)v20;
        v22 = (v21 - (char *)v17) >> 3;
        if ( v21 - (char *)v17 > 0 )
        {
          while ( 1 )
          {
            v23 = *((_QWORD *)v21 - 1);
            v24 = *(_QWORD *)v21;
            *((_QWORD *)v21 - 1) = 0;
            *(_QWORD *)v21 = v23;
            if ( v24 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
            if ( !--v22 )
              break;
            v21 -= 8;
          }
        }
        v25 = v41[0];
        v41[0] = 0;
        v26 = *v17;
        *v17 = v25;
        if ( v26 )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v26 + 8LL))(v26);
      }
      v16 = v41[0];
      goto LABEL_31;
    }
  }
  v8 = *a2;
  *a2 = 0;
  v40 = v8 & 0xFFFFFFFFFFFFFFFELL;
  v9 = *a3;
  *a3 = 0;
  v41[0] = v9 & 0xFFFFFFFFFFFFFFFELL;
  v10 = (__int64 *)sub_22077B0(32);
  v11 = (unsigned __int64)v10;
  if ( !v10 )
    goto LABEL_58;
  v10[1] = 0;
  v10[2] = 0;
  v10[3] = 0;
  *v10 = (__int64)&unk_49DC7A0;
  sub_9CD990(v10 + 1, 0, &v40);
  v12 = *(char **)(v11 + 16);
  if ( *(char **)(v11 + 24) == v12 )
  {
    sub_9CD990((__int64 *)(v11 + 8), v12, v41);
LABEL_58:
    v37 = v41[0];
LABEL_52:
    *a1 = v11 | 1;
    if ( v37 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v37 + 8LL))(v37);
    goto LABEL_15;
  }
  if ( !v12 )
  {
    *(_QWORD *)(v11 + 16) = 8;
    v37 = v41[0];
    goto LABEL_52;
  }
  *(_QWORD *)v12 = v41[0];
  *(_QWORD *)(v11 + 16) += 8LL;
  *a1 = v11 | 1;
LABEL_15:
  if ( v40 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v40 + 8LL))(v40);
  return a1;
}
