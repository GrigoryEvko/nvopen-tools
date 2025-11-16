// Function: sub_38E0040
// Address: 0x38e0040
//
void __fastcall sub_38E0040(__int64 a1, const char **a2)
{
  __int64 v2; // r14
  char v3; // bl
  void (**v4)(); // rax
  void (*v5)(); // rax
  __int64 v6; // rdx
  _DWORD *v7; // rax
  _DWORD *i; // rcx
  __int64 v9; // r13
  bool v10; // zf
  unsigned __int64 v11; // r12
  __int64 v12; // rbx
  unsigned __int64 v13; // r12
  signed __int64 v14; // r15
  char *v15; // rcx
  __int64 v16; // r15
  size_t v17; // rax
  __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  __int64 v20; // [rsp+8h] [rbp-88h]
  const char *v21; // [rsp+10h] [rbp-80h] BYREF
  __int64 v22; // [rsp+18h] [rbp-78h]
  __int64 v23; // [rsp+20h] [rbp-70h]
  __int64 v24; // [rsp+28h] [rbp-68h]
  unsigned __int64 v25; // [rsp+30h] [rbp-60h]
  __int64 v26; // [rsp+38h] [rbp-58h]
  __int64 v27; // [rsp+40h] [rbp-50h]
  __int64 v28; // [rsp+48h] [rbp-48h]
  __int64 v29; // [rsp+50h] [rbp-40h]
  char v30; // [rsp+58h] [rbp-38h]
  char v31; // [rsp+59h] [rbp-37h]
  int v32; // [rsp+5Ch] [rbp-34h]

  v2 = a1;
  v3 = (char)a2;
  if ( sub_38DD120(a1) )
  {
    a1 = *(_QWORD *)(a1 + 8);
    a2 = 0;
    LOWORD(v23) = 259;
    v21 = "starting new .cfi frame before finishing the previous one";
    sub_38BE3D0(a1, 0, (__int64)&v21);
  }
  v4 = *(void (***)())v2;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v32 = 0x7FFFFFFF;
  v31 = v3;
  v5 = *v4;
  if ( v5 != nullsub_1937 )
  {
    a2 = &v21;
    a1 = v2;
    ((void (__fastcall *)(__int64, const char **))v5)(v2, &v21);
  }
  v6 = *(_QWORD *)(*(_QWORD *)(v2 + 8) + 16LL);
  if ( v6 )
  {
    v7 = *(_DWORD **)(v6 + 368);
    for ( i = *(_DWORD **)(v6 + 376); i != v7; v7 += 12 )
    {
      v6 = *v7 & 0xFFFFFFFD;
      if ( (_DWORD)v6 == 4 )
      {
        v6 = (unsigned int)v7[4];
        LODWORD(v28) = v7[4];
      }
    }
  }
  v9 = *(_QWORD *)(v2 + 32);
  if ( v9 == *(_QWORD *)(v2 + 40) )
  {
    sub_38DFBA0(v2 + 24, *(char **)(v2 + 32), (__int64 *)&v21);
    v13 = v25;
  }
  else
  {
    if ( v9 )
    {
      *(_QWORD *)v9 = v21;
      *(_QWORD *)(v9 + 8) = v22;
      *(_QWORD *)(v9 + 16) = v23;
      *(_QWORD *)(v9 + 24) = v24;
      v11 = v26 - v25;
      v10 = v26 == v25;
      *(_QWORD *)(v9 + 32) = 0;
      *(_QWORD *)(v9 + 40) = 0;
      *(_QWORD *)(v9 + 48) = 0;
      if ( v10 )
      {
        v12 = 0;
      }
      else
      {
        if ( v11 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_40:
          sub_4261EA(a1, a2, v6);
        a1 = v11;
        v12 = sub_22077B0(v11);
      }
      *(_QWORD *)(v9 + 32) = v12;
      *(_QWORD *)(v9 + 40) = v12;
      *(_QWORD *)(v9 + 48) = v12 + v11;
      v13 = v25;
      v20 = v26;
      if ( v26 != v25 )
      {
        do
        {
          if ( v12 )
          {
            *(_DWORD *)v12 = *(_DWORD *)v13;
            *(_QWORD *)(v12 + 8) = *(_QWORD *)(v13 + 8);
            *(_DWORD *)(v12 + 16) = *(_DWORD *)(v13 + 16);
            *(_DWORD *)(v12 + 20) = *(_DWORD *)(v13 + 20);
            v14 = *(_QWORD *)(v13 + 32) - *(_QWORD *)(v13 + 24);
            *(_QWORD *)(v12 + 24) = 0;
            *(_QWORD *)(v12 + 32) = 0;
            *(_QWORD *)(v12 + 40) = 0;
            if ( v14 )
            {
              if ( v14 < 0 )
                goto LABEL_40;
              a1 = v14;
              v15 = (char *)sub_22077B0(v14);
            }
            else
            {
              v15 = 0;
            }
            *(_QWORD *)(v12 + 24) = v15;
            *(_QWORD *)(v12 + 40) = &v15[v14];
            v16 = 0;
            *(_QWORD *)(v12 + 32) = v15;
            a2 = *(const char ***)(v13 + 24);
            v17 = *(_QWORD *)(v13 + 32) - (_QWORD)a2;
            if ( v17 )
            {
              a1 = (__int64)v15;
              v16 = *(_QWORD *)(v13 + 32) - (_QWORD)a2;
              v15 = (char *)memmove(v15, a2, v17);
            }
            *(_QWORD *)(v12 + 32) = &v15[v16];
          }
          v12 += 48;
          v13 += 48LL;
        }
        while ( v20 != v13 );
        v13 = v25;
      }
      *(_QWORD *)(v9 + 40) = v12;
      *(_QWORD *)(v9 + 56) = v28;
      *(_QWORD *)(v9 + 64) = v29;
      *(_BYTE *)(v9 + 72) = v30;
      *(_BYTE *)(v9 + 73) = v31;
      *(_DWORD *)(v9 + 76) = v32;
      v9 = *(_QWORD *)(v2 + 32);
    }
    else
    {
      v13 = v25;
    }
    *(_QWORD *)(v2 + 32) = v9 + 80;
  }
  v18 = v26;
  if ( v26 != v13 )
  {
    do
    {
      v19 = *(_QWORD *)(v13 + 24);
      if ( v19 )
        j_j___libc_free_0(v19);
      v13 += 48LL;
    }
    while ( v18 != v13 );
    v13 = v25;
  }
  if ( v13 )
    j_j___libc_free_0(v13);
}
