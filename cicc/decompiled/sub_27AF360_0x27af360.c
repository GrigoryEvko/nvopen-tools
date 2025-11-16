// Function: sub_27AF360
// Address: 0x27af360
//
void __fastcall sub_27AF360(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v5; // r15
  int v6; // edx
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // edx
  unsigned int v11; // esi
  __int64 *v12; // rax
  __int64 v13; // r11
  unsigned int v14; // r11d
  unsigned int v15; // esi
  __int64 *v16; // rax
  __int64 v17; // r10
  __int64 v18; // r12
  __int64 v19; // r8
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // eax
  unsigned int v24; // r10d
  __int64 v25; // r14
  void *v26; // r8
  int v27; // eax
  int v28; // ecx
  int v29; // ecx
  unsigned int v30; // [rsp+4h] [rbp-7Ch]
  __int64 v31; // [rsp+8h] [rbp-78h]
  void *src; // [rsp+18h] [rbp-68h]
  void *srca; // [rsp+18h] [rbp-68h]
  _QWORD *v35; // [rsp+28h] [rbp-58h]
  __int64 v36; // [rsp+30h] [rbp-50h] BYREF
  void *v37; // [rsp+38h] [rbp-48h]
  __int64 v38; // [rsp+40h] [rbp-40h]
  unsigned int v39; // [rsp+48h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 2 )
  {
    v5 = a1 + 4;
    do
    {
      v6 = *(_DWORD *)(a3 + 24);
      v7 = *(v5 - 2);
      v35 = v5;
      v8 = *a1;
      v9 = *(_QWORD *)(a3 + 8);
      if ( v6 )
      {
        v10 = v6 - 1;
        v11 = v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v12 = (__int64 *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( v7 == *v12 )
        {
LABEL_6:
          v14 = *((_DWORD *)v12 + 2);
        }
        else
        {
          v27 = 1;
          while ( v13 != -4096 )
          {
            v29 = v27 + 1;
            v11 = v10 & (v27 + v11);
            v12 = (__int64 *)(v9 + 16LL * v11);
            v13 = *v12;
            if ( v7 == *v12 )
              goto LABEL_6;
            v27 = v29;
          }
          v14 = 0;
        }
        v15 = v10 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v16 = (__int64 *)(v9 + 16LL * v15);
        v17 = *v16;
        if ( v8 == *v16 )
        {
LABEL_8:
          if ( v14 < *((_DWORD *)v16 + 2) )
          {
            v18 = (char *)(v5 - 2) - (char *)a1;
            v19 = *(v5 - 1);
            v20 = v5;
            v21 = v18 >> 4;
            if ( v18 > 0 )
            {
              do
              {
                v22 = *(v20 - 4);
                v20 -= 2;
                *v20 = v22;
                v20[1] = *(v20 - 1);
                --v21;
              }
              while ( v21 );
            }
            *a1 = v7;
            a1[1] = v19;
            goto LABEL_12;
          }
        }
        else
        {
          v23 = 1;
          while ( v17 != -4096 )
          {
            v28 = v23 + 1;
            v15 = v10 & (v23 + v15);
            v16 = (__int64 *)(v9 + 16LL * v15);
            v17 = *v16;
            if ( v8 == *v16 )
              goto LABEL_8;
            v23 = v28;
          }
        }
      }
      sub_C7D6A0(0, 0, 8);
      v24 = *(_DWORD *)(a3 + 24);
      if ( v24 )
      {
        v30 = *(_DWORD *)(a3 + 24);
        v25 = 16LL * v24;
        src = (void *)sub_C7D670(v25, 8);
        v31 = *(_QWORD *)(a3 + 16);
        memcpy(src, *(const void **)(a3 + 8), v25);
        v36 = 0;
        v37 = 0;
        v38 = 0;
        v39 = 0;
        sub_C7D6A0(0, 0, 8);
        v39 = v30;
        v37 = (void *)sub_C7D670(v25, 8);
        v38 = v31;
        memcpy(v37, src, 16LL * v39);
        v26 = src;
      }
      else
      {
        v39 = 0;
        v36 = 0;
        v25 = 0;
        v37 = 0;
        v38 = 0;
        sub_C7D6A0(0, 0, 8);
        v37 = 0;
        v26 = 0;
        v38 = 0;
        v39 = 0;
      }
      srca = v26;
      sub_27ACAE0(v5 - 2, (__int64)&v36);
      sub_C7D6A0((__int64)v37, 16LL * v39, 8);
      sub_C7D6A0((__int64)srca, v25, 8);
LABEL_12:
      v5 += 2;
    }
    while ( a2 != v35 );
  }
}
