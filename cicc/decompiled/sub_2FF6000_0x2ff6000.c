// Function: sub_2FF6000
// Address: 0x2ff6000
//
void __fastcall sub_2FF6000(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdi
  bool v6; // zf
  const char *v7; // rdi
  size_t v8; // rax
  _BYTE *v9; // rax
  unsigned __int64 v10; // rax
  _QWORD v11[2]; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int8 *v12[2]; // [rsp+10h] [rbp-30h] BYREF
  __int64 v13; // [rsp+20h] [rbp-20h] BYREF

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 56LL) + 16LL * (**(_DWORD **)a1 & 0x7FFFFFFF));
  if ( v2 )
  {
    if ( ((v2 >> 2) & 1) != 0 )
    {
      if ( ((v2 >> 2) & 1) != 0 )
      {
        v10 = v2 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v10 )
        {
          v7 = *(const char **)(v10 + 8);
          v8 = 0;
          v11[0] = v7;
          if ( !v7 )
          {
LABEL_6:
            v11[1] = v8;
            sub_C93130((__int64 *)v12, (__int64)v11);
            sub_CB6200(a2, v12[0], (size_t)v12[1]);
            if ( (__int64 *)v12[0] != &v13 )
              j_j___libc_free_0((unsigned __int64)v12[0]);
            return;
          }
LABEL_5:
          v8 = strlen(v7);
          goto LABEL_6;
        }
      }
    }
    else
    {
      v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v3 )
      {
        v4 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
        v5 = *(unsigned int *)(*(_QWORD *)v3 + 16LL);
        v6 = *(_QWORD *)(v4 + 80) + v5 == 0;
        v7 = (const char *)(*(_QWORD *)(v4 + 80) + v5);
        v8 = 0;
        v11[0] = v7;
        if ( v6 )
          goto LABEL_6;
        goto LABEL_5;
      }
    }
  }
  v9 = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == v9 )
  {
    sub_CB6200(a2, (unsigned __int8 *)"_", 1u);
  }
  else
  {
    *v9 = 95;
    ++*(_QWORD *)(a2 + 32);
  }
}
