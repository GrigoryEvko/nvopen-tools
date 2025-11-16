// Function: sub_1E60B80
// Address: 0x1e60b80
//
__int64 __fastcall sub_1E60B80(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned __int8 (__fastcall *a4)(__int64),
        int a5)
{
  __int64 v5; // r15
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 *v10; // rax
  int v11; // r9d
  __int64 v12; // r8
  __int64 v13; // rcx
  _BYTE *v14; // rsi
  int v15; // r8d
  __int64 *v16; // r12
  __int64 *v17; // r13
  __int64 *v18; // rax
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // r9d
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // r10
  int v30; // edi
  __int64 *v31; // [rsp+8h] [rbp-2E8h]
  __int64 v34; // [rsp+48h] [rbp-2A8h] BYREF
  __int64 v35; // [rsp+50h] [rbp-2A0h] BYREF
  __int64 v36; // [rsp+58h] [rbp-298h] BYREF
  __int64 *v37; // [rsp+60h] [rbp-290h] BYREF
  int v38; // [rsp+68h] [rbp-288h]
  char v39; // [rsp+70h] [rbp-280h] BYREF
  _QWORD *v40; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v41; // [rsp+B8h] [rbp-238h]
  _QWORD v42[70]; // [rsp+C0h] [rbp-230h] BYREF

  v5 = a1 + 24;
  v34 = a2;
  v40 = v42;
  v42[0] = a2;
  v36 = a2;
  v41 = 0x4000000001LL;
  if ( (unsigned __int8)sub_1E5F140(a1 + 24, &v36, &v37) )
    *((_DWORD *)sub_1E60050(v5, &v34) + 3) = a5;
  v8 = v41;
  if ( (_DWORD)v41 )
  {
    while ( 1 )
    {
      v9 = v40[v8 - 1];
      LODWORD(v41) = v8 - 1;
      v35 = v9;
      v10 = sub_1E60050(v5, &v35);
      if ( *((_DWORD *)v10 + 2) )
        goto LABEL_5;
      ++a3;
      v12 = v35;
      v13 = a3;
      v10[3] = v35;
      *((_DWORD *)v10 + 4) = a3;
      *((_DWORD *)v10 + 2) = a3;
      v14 = *(_BYTE **)(a1 + 8);
      if ( v14 == *(_BYTE **)(a1 + 16) )
      {
        sub_1D4AF10(a1, v14, &v35);
        v12 = v35;
      }
      else
      {
        if ( v14 )
        {
          *(_QWORD *)v14 = v12;
          v14 = *(_BYTE **)(a1 + 8);
          v12 = v35;
        }
        *(_QWORD *)(a1 + 8) = v14 + 8;
      }
      sub_1E5F7F0((__int64)&v37, v12, *(_QWORD *)(a1 + 56), v13, v12, v11);
      v15 = (int)v37;
      v16 = &v37[v38];
      if ( v37 != v16 )
      {
        v17 = v37;
        while ( 1 )
        {
          v22 = *v17;
          v23 = *(unsigned int *)(a1 + 48);
          v36 = *v17;
          if ( !(_DWORD)v23 )
            goto LABEL_13;
          v24 = v23 - 1;
          v25 = *(_QWORD *)(a1 + 32);
          v26 = ((_DWORD)v23 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v27 = v25 + 72 * v26;
          v28 = *(_QWORD *)v27;
          if ( v22 != *(_QWORD *)v27 )
            break;
LABEL_20:
          if ( v27 == v25 + 72 * v23 || !*(_DWORD *)(v27 + 8) )
            goto LABEL_13;
          if ( v22 == v35 )
          {
LABEL_17:
            if ( v16 == ++v17 )
              goto LABEL_24;
          }
          else
          {
            ++v17;
            sub_1E05890(v27 + 40, &v35, v26, v25, v15, v24);
            if ( v16 == v17 )
            {
LABEL_24:
              v16 = v37;
              goto LABEL_25;
            }
          }
        }
        v30 = 1;
        while ( v28 != -8 )
        {
          v15 = v30 + 1;
          v26 = v24 & (unsigned int)(v30 + v26);
          v27 = v25 + 72LL * (unsigned int)v26;
          v28 = *(_QWORD *)v27;
          if ( v22 == *(_QWORD *)v27 )
            goto LABEL_20;
          v30 = v15;
        }
LABEL_13:
        if ( a4(v35) )
        {
          v18 = sub_1E60050(v5, &v36);
          v21 = (unsigned int)v41;
          if ( (unsigned int)v41 >= HIDWORD(v41) )
          {
            v31 = v18;
            sub_16CD150((__int64)&v40, v42, 0, 8, v19, v20);
            v21 = (unsigned int)v41;
            v18 = v31;
          }
          v40[v21] = v36;
          LODWORD(v41) = v41 + 1;
          *((_DWORD *)v18 + 3) = a3;
          sub_1E05890((__int64)(v18 + 5), &v35, v21, a3, v19, v20);
        }
        goto LABEL_17;
      }
LABEL_25:
      if ( v16 == (__int64 *)&v39 )
      {
LABEL_5:
        v8 = v41;
        if ( !(_DWORD)v41 )
          break;
      }
      else
      {
        _libc_free((unsigned __int64)v16);
        v8 = v41;
        if ( !(_DWORD)v41 )
          break;
      }
    }
  }
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  return a3;
}
