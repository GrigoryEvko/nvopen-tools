// Function: sub_1CB9110
// Address: 0x1cb9110
//
_QWORD *__fastcall sub_1CB9110(__int64 *a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // r12
  int v3; // eax
  __int64 v4; // rax
  __int64 v6; // r10
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 *v14; // r12
  size_t v15; // r15
  __int64 v16; // r8
  void *v17; // rdi
  void *v18; // rax
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+10h] [rbp-60h]
  _QWORD *v23; // [rsp+20h] [rbp-50h]
  __int64 v24; // [rsp+28h] [rbp-48h]
  unsigned __int64 v25; // [rsp+30h] [rbp-40h]
  __int64 v26; // [rsp+30h] [rbp-40h]
  unsigned __int64 v27; // [rsp+38h] [rbp-38h]
  __int64 v28; // [rsp+38h] [rbp-38h]

  v1 = (_QWORD *)sub_22077B0(192);
  v2 = v1;
  if ( v1 )
  {
    v1[1] = 0;
    v1[2] = &unk_4FBEC7C;
    v1[10] = v1 + 8;
    v1[11] = v1 + 8;
    v1[16] = v1 + 14;
    v1[17] = v1 + 14;
    *v1 = off_49F8750;
    v1[22] = 0x1000000000LL;
    v3 = *((_DWORD *)a1 + 3);
    *((_DWORD *)v2 + 6) = 5;
    v2[4] = 0;
    v2[5] = 0;
    v2[6] = 0;
    *((_DWORD *)v2 + 16) = 0;
    v2[9] = 0;
    v2[12] = 0;
    *((_DWORD *)v2 + 28) = 0;
    v2[15] = 0;
    v2[18] = 0;
    *((_BYTE *)v2 + 152) = 0;
    v2[20] = 0;
    v2[21] = 0;
    if ( v3 )
    {
      sub_16D1890((__int64)(v2 + 20), *((_DWORD *)a1 + 2));
      v6 = v2[20];
      v7 = *a1;
      v8 = *((unsigned int *)v2 + 42);
      v9 = 8 * v8 + 8;
      v21 = v6;
      v20 = *a1;
      *(_QWORD *)((char *)v2 + 172) = *(__int64 *)((char *)a1 + 12);
      if ( (_DWORD)v8 )
      {
        v23 = v2;
        v24 = 8LL * (unsigned int)(v8 - 1);
        v10 = 0;
        v11 = v7;
        v12 = v9;
        while ( 1 )
        {
          v13 = *(_QWORD *)(v11 + v10);
          v14 = (__int64 *)(v6 + v10);
          if ( v13 != -8 )
          {
            if ( v13 )
              break;
          }
          *v14 = v13;
          v12 += 4;
          if ( v10 == v24 )
          {
LABEL_15:
            v2 = v23;
            goto LABEL_3;
          }
LABEL_8:
          v10 += 8;
          v11 = *a1;
          v6 = v23[20];
        }
        v15 = *(_QWORD *)v13;
        v25 = *(_QWORD *)v13 + 17LL;
        v27 = *(_QWORD *)v13 + 1LL;
        v16 = malloc(v25);
        if ( !v16 )
        {
          if ( !v25 )
          {
            v19 = malloc(1u);
            v16 = 0;
            if ( v19 )
            {
              v17 = (void *)(v19 + 16);
              v16 = v19;
LABEL_13:
              v28 = v16;
              v18 = memcpy(v17, (const void *)(v13 + 16), v15);
              v16 = v28;
              v17 = v18;
LABEL_7:
              *((_BYTE *)v17 + v15) = 0;
              *(_QWORD *)v16 = v15;
              *(_DWORD *)(v16 + 8) = *(_DWORD *)(v13 + 8);
              *v14 = v16;
              *(_DWORD *)(v21 + v12) = *(_DWORD *)(v20 + v12);
              v12 += 4;
              if ( v10 == v24 )
                goto LABEL_15;
              goto LABEL_8;
            }
          }
          v26 = v16;
          sub_16BD1C0("Allocation failed", 1u);
          v16 = v26;
        }
        v17 = (void *)(v16 + 16);
        if ( v27 <= 1 )
          goto LABEL_7;
        goto LABEL_13;
      }
    }
LABEL_3:
    v4 = sub_163A1D0();
    sub_1CB8F20(v4);
  }
  return v2;
}
