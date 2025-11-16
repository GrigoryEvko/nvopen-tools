// Function: sub_38DF630
// Address: 0x38df630
//
void __fastcall sub_38DF630(__int64 a1, __int64 a2)
{
  __int64 (*v2)(void); // rdx
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r13
  signed __int64 v9; // r13
  char *v10; // rcx
  size_t v11; // r14
  __int64 v12; // rax
  void *v13; // r13
  size_t v14; // rax
  int v15; // [rsp+10h] [rbp-50h] BYREF
  __int64 v16; // [rsp+18h] [rbp-48h]
  int v17; // [rsp+20h] [rbp-40h]
  int v18; // [rsp+24h] [rbp-3Ch]
  void *src; // [rsp+28h] [rbp-38h]
  __int64 v20; // [rsp+30h] [rbp-30h]
  __int64 v21; // [rsp+38h] [rbp-28h]

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 16LL);
  v3 = 1;
  if ( v2 != sub_38DBC10 )
    v3 = v2();
  v17 = a2;
  v15 = 0;
  v16 = v3;
  v18 = 0;
  src = 0;
  v20 = 0;
  v21 = 0;
  v4 = sub_38DD140(a1);
  v6 = v4;
  if ( v4 )
  {
    v7 = *(_QWORD *)(v4 + 40);
    if ( v7 == *(_QWORD *)(v4 + 48) )
    {
      sub_20E34D0(v4 + 32, *(char **)(v4 + 40), &v15);
      v13 = src;
    }
    else
    {
      if ( v7 )
      {
        *(_QWORD *)(v7 + 40) = 0;
        *(_DWORD *)v7 = v15;
        *(_QWORD *)(v7 + 8) = v16;
        *(_DWORD *)(v7 + 16) = v17;
        *(_DWORD *)(v7 + 20) = v18;
        v8 = v20;
        *(_QWORD *)(v7 + 32) = 0;
        v9 = v8 - (_QWORD)src;
        *(_QWORD *)(v7 + 24) = 0;
        if ( v9 )
        {
          if ( v9 < 0 )
            sub_4261EA(a1, a2, v5);
          v10 = (char *)sub_22077B0(v9);
        }
        else
        {
          v10 = 0;
        }
        *(_QWORD *)(v7 + 24) = v10;
        v11 = 0;
        *(_QWORD *)(v7 + 32) = v10;
        v12 = v20;
        *(_QWORD *)(v7 + 40) = &v10[v9];
        v13 = src;
        v14 = v12 - (_QWORD)src;
        if ( v14 )
        {
          v11 = v14;
          v10 = (char *)memmove(v10, src, v14);
        }
        *(_QWORD *)(v7 + 32) = &v10[v11];
        v7 = *(_QWORD *)(v6 + 40);
      }
      else
      {
        v13 = src;
      }
      *(_QWORD *)(v6 + 40) = v7 + 48;
    }
    if ( v13 )
      j_j___libc_free_0((unsigned __int64)v13);
  }
}
