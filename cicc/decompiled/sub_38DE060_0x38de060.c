// Function: sub_38DE060
// Address: 0x38de060
//
void __fastcall sub_38DE060(__int64 a1, const void *a2, signed __int64 a3)
{
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  char *v6; // rbx
  char *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // r12
  char *v12; // r13
  signed __int64 v13; // r13
  char *v14; // rcx
  size_t v15; // r14
  char *v16; // rax
  void *v17; // r13
  size_t v18; // rax
  int v19; // [rsp+0h] [rbp-50h] BYREF
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  void *src; // [rsp+18h] [rbp-38h]
  char *v23; // [rsp+20h] [rbp-30h]
  char *v24; // [rsp+28h] [rbp-28h]

  v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 16LL);
  v5 = 1;
  if ( v4 != sub_38DBC10 )
    v5 = v4();
  v19 = 9;
  v20 = v5;
  v21 = 0;
  src = 0;
  v23 = 0;
  v24 = 0;
  if ( a3 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v6 = 0;
  if ( a3 )
  {
    v7 = (char *)sub_22077B0(a3);
    v6 = &v7[a3];
    src = v7;
    v24 = &v7[a3];
    memcpy(v7, a2, a3);
  }
  v23 = v6;
  v8 = sub_38DD140(a1);
  v10 = v8;
  if ( v8 )
  {
    v11 = *(_QWORD *)(v8 + 40);
    if ( v11 == *(_QWORD *)(v8 + 48) )
    {
      sub_20E34D0(v8 + 32, *(char **)(v8 + 40), &v19);
      v17 = src;
    }
    else
    {
      if ( v11 )
      {
        *(_QWORD *)(v11 + 40) = 0;
        *(_DWORD *)v11 = v19;
        *(_QWORD *)(v11 + 8) = v20;
        *(_QWORD *)(v11 + 16) = v21;
        v12 = v23;
        *(_QWORD *)(v11 + 32) = 0;
        v13 = v12 - (_BYTE *)src;
        *(_QWORD *)(v11 + 24) = 0;
        if ( v13 )
        {
          if ( v13 < 0 )
            sub_4261EA(a1, a2, v9);
          v14 = (char *)sub_22077B0(v13);
        }
        else
        {
          v14 = 0;
        }
        *(_QWORD *)(v11 + 24) = v14;
        v15 = 0;
        *(_QWORD *)(v11 + 32) = v14;
        v16 = v23;
        *(_QWORD *)(v11 + 40) = &v14[v13];
        v17 = src;
        v18 = v16 - (_BYTE *)src;
        if ( v18 )
        {
          v15 = v18;
          v14 = (char *)memmove(v14, src, v18);
        }
        *(_QWORD *)(v11 + 32) = &v14[v15];
        v11 = *(_QWORD *)(v10 + 40);
      }
      else
      {
        v17 = src;
      }
      *(_QWORD *)(v10 + 40) = v11 + 48;
    }
    if ( v17 )
      j_j___libc_free_0((unsigned __int64)v17);
  }
  else if ( src )
  {
    j_j___libc_free_0((unsigned __int64)src);
  }
}
