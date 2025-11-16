// Function: sub_12F5270
// Address: 0x12f5270
//
void __fastcall sub_12F5270(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r15
  __int64 v4; // r14
  size_t v5; // rdx
  size_t v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // r10
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned int v15; // r9d
  __int64 *v16; // r10
  __int64 v17; // r8
  void *v18; // rdi
  __int64 v19; // rax
  void *v20; // rax
  __int64 *v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  unsigned int v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  __int64 *v25; // [rsp+10h] [rbp-50h]
  __int64 *v26; // [rsp+18h] [rbp-48h]
  unsigned int v27; // [rsp+18h] [rbp-48h]
  void *src; // [rsp+20h] [rbp-40h]
  unsigned int v29; // [rsp+28h] [rbp-38h]

  v2 = a1 + 24;
  v3 = *(_QWORD *)(a1 + 32);
  if ( v3 != a1 + 24 )
  {
    while ( 1 )
    {
      v4 = v3 - 56;
      if ( !v3 )
        v4 = 0;
      if ( (unsigned __int8)sub_15E4F60(v4) || (*(_BYTE *)(v4 + 32) & 0xF) != 0 )
        goto LABEL_3;
      src = (void *)sub_1649960(v4);
      v6 = v5;
      v10 = (unsigned int)sub_16D19C0(a2, src, v5);
      v11 = (__int64 *)(*(_QWORD *)a2 + 8 * v10);
      if ( !*v11 )
        goto LABEL_11;
      if ( *v11 != -8 )
      {
LABEL_3:
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return;
      }
      else
      {
        --*(_DWORD *)(a2 + 16);
LABEL_11:
        v21 = v11;
        v23 = v10;
        v12 = malloc(v6 + 17, v6 + 17, v7, v8, v9, v10);
        v15 = v23;
        v16 = v21;
        v17 = v12;
        if ( !v12 )
        {
          if ( v6 == -17 )
          {
            v19 = malloc(1, 0, v13, v14, 0, v23);
            v15 = v23;
            v16 = v21;
            v17 = 0;
            if ( v19 )
            {
              v18 = (void *)(v19 + 16);
              v17 = v19;
LABEL_17:
              v24 = v17;
              v26 = v16;
              v29 = v15;
              v20 = memcpy(v18, src, v6);
              v17 = v24;
              v16 = v26;
              v15 = v29;
              v18 = v20;
              goto LABEL_13;
            }
          }
          v22 = v17;
          v25 = v16;
          v27 = v15;
          sub_16BD1C0("Allocation failed");
          v15 = v27;
          v16 = v25;
          v17 = v22;
        }
        v18 = (void *)(v17 + 16);
        if ( v6 + 1 > 1 )
          goto LABEL_17;
LABEL_13:
        *((_BYTE *)v18 + v6) = 0;
        *(_QWORD *)v17 = v6;
        *(_BYTE *)(v17 + 8) = 0;
        *v16 = v17;
        ++*(_DWORD *)(a2 + 12);
        sub_16D1CD0(a2, v15);
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return;
      }
    }
  }
}
