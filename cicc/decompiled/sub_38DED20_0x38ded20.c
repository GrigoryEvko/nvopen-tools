// Function: sub_38DED20
// Address: 0x38ded20
//
void __fastcall sub_38DED20(__int64 a1, __int64 a2, int a3)
{
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // r13
  signed __int64 v11; // r13
  char *v12; // rcx
  size_t v13; // r14
  __int64 v14; // rax
  void *v15; // r13
  size_t v16; // rax
  int v17; // [rsp+10h] [rbp-50h] BYREF
  __int64 v18; // [rsp+18h] [rbp-48h]
  int v19; // [rsp+20h] [rbp-40h]
  int v20; // [rsp+24h] [rbp-3Ch]
  void *src; // [rsp+28h] [rbp-38h]
  __int64 v22; // [rsp+30h] [rbp-30h]
  __int64 v23; // [rsp+38h] [rbp-28h]

  v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 16LL);
  v5 = 1;
  if ( v4 != sub_38DBC10 )
    v5 = v4();
  v20 = a3;
  v17 = 3;
  v18 = v5;
  v19 = a2;
  src = 0;
  v22 = 0;
  v23 = 0;
  v6 = sub_38DD140(a1);
  v8 = v6;
  if ( v6 )
  {
    v9 = *(_QWORD *)(v6 + 40);
    if ( v9 == *(_QWORD *)(v6 + 48) )
    {
      sub_20E34D0(v6 + 32, *(char **)(v6 + 40), &v17);
      v15 = src;
    }
    else
    {
      if ( v9 )
      {
        *(_QWORD *)(v9 + 40) = 0;
        *(_DWORD *)v9 = v17;
        *(_QWORD *)(v9 + 8) = v18;
        *(_DWORD *)(v9 + 16) = v19;
        *(_DWORD *)(v9 + 20) = v20;
        v10 = v22;
        *(_QWORD *)(v9 + 32) = 0;
        v11 = v10 - (_QWORD)src;
        *(_QWORD *)(v9 + 24) = 0;
        if ( v11 )
        {
          if ( v11 < 0 )
            sub_4261EA(a1, a2, v7);
          v12 = (char *)sub_22077B0(v11);
        }
        else
        {
          v12 = 0;
        }
        *(_QWORD *)(v9 + 24) = v12;
        v13 = 0;
        *(_QWORD *)(v9 + 32) = v12;
        v14 = v22;
        *(_QWORD *)(v9 + 40) = &v12[v11];
        v15 = src;
        v16 = v14 - (_QWORD)src;
        if ( v16 )
        {
          v13 = v16;
          v12 = (char *)memmove(v12, src, v16);
        }
        *(_QWORD *)(v9 + 32) = &v12[v13];
        v9 = *(_QWORD *)(v8 + 40);
      }
      else
      {
        v15 = src;
      }
      *(_QWORD *)(v8 + 40) = v9 + 48;
    }
    if ( v15 )
      j_j___libc_free_0((unsigned __int64)v15);
  }
}
