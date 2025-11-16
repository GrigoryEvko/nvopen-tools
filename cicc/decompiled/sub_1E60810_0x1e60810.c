// Function: sub_1E60810
// Address: 0x1e60810
//
__int64 __fastcall sub_1E60810(
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
  __int64 v11; // r8
  _BYTE *v12; // rsi
  int v13; // r8d
  __int64 *v14; // r12
  __int64 *v15; // r13
  __int64 *v16; // rax
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // r9d
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // r10
  int v28; // edi
  __int64 *v29; // [rsp+8h] [rbp-2E8h]
  __int64 v32; // [rsp+48h] [rbp-2A8h] BYREF
  __int64 v33; // [rsp+50h] [rbp-2A0h] BYREF
  __int64 v34; // [rsp+58h] [rbp-298h] BYREF
  __int64 *v35; // [rsp+60h] [rbp-290h] BYREF
  int v36; // [rsp+68h] [rbp-288h]
  char v37; // [rsp+70h] [rbp-280h] BYREF
  _QWORD *v38; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v39; // [rsp+B8h] [rbp-238h]
  _QWORD v40[70]; // [rsp+C0h] [rbp-230h] BYREF

  v5 = a1 + 24;
  v32 = a2;
  v38 = v40;
  v40[0] = a2;
  v34 = a2;
  v39 = 0x4000000001LL;
  if ( (unsigned __int8)sub_1E5F140(a1 + 24, &v34, &v35) )
    *((_DWORD *)sub_1E60050(v5, &v32) + 3) = a5;
  v8 = v39;
  if ( (_DWORD)v39 )
  {
    while ( 1 )
    {
      v9 = v38[v8 - 1];
      LODWORD(v39) = v8 - 1;
      v33 = v9;
      v10 = sub_1E60050(v5, &v33);
      if ( *((_DWORD *)v10 + 2) )
        goto LABEL_5;
      ++a3;
      v11 = v33;
      v10[3] = v33;
      *((_DWORD *)v10 + 4) = a3;
      *((_DWORD *)v10 + 2) = a3;
      v12 = *(_BYTE **)(a1 + 8);
      if ( v12 == *(_BYTE **)(a1 + 16) )
      {
        sub_1D4AF10(a1, v12, &v33);
        v11 = v33;
      }
      else
      {
        if ( v12 )
        {
          *(_QWORD *)v12 = v11;
          v12 = *(_BYTE **)(a1 + 8);
          v11 = v33;
        }
        *(_QWORD *)(a1 + 8) = v12 + 8;
      }
      sub_1E5F5F0((__int64)&v35, v11, *(_QWORD *)(a1 + 56));
      v13 = (int)v35;
      v14 = &v35[v36];
      if ( v35 != v14 )
      {
        v15 = v35;
        while ( 1 )
        {
          v20 = *v15;
          v21 = *(unsigned int *)(a1 + 48);
          v34 = *v15;
          if ( !(_DWORD)v21 )
            goto LABEL_13;
          v22 = v21 - 1;
          v23 = *(_QWORD *)(a1 + 32);
          v24 = ((_DWORD)v21 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v25 = v23 + 72 * v24;
          v26 = *(_QWORD *)v25;
          if ( v20 != *(_QWORD *)v25 )
            break;
LABEL_20:
          if ( v25 == v23 + 72 * v21 || !*(_DWORD *)(v25 + 8) )
            goto LABEL_13;
          if ( v20 == v33 )
          {
LABEL_17:
            if ( v14 == ++v15 )
              goto LABEL_24;
          }
          else
          {
            ++v15;
            sub_1E05890(v25 + 40, &v33, v24, v23, v13, v22);
            if ( v14 == v15 )
            {
LABEL_24:
              v14 = v35;
              goto LABEL_25;
            }
          }
        }
        v28 = 1;
        while ( v26 != -8 )
        {
          v13 = v28 + 1;
          v24 = v22 & (unsigned int)(v28 + v24);
          v25 = v23 + 72LL * (unsigned int)v24;
          v26 = *(_QWORD *)v25;
          if ( v20 == *(_QWORD *)v25 )
            goto LABEL_20;
          v28 = v13;
        }
LABEL_13:
        if ( a4(v33) )
        {
          v16 = sub_1E60050(v5, &v34);
          v19 = (unsigned int)v39;
          if ( (unsigned int)v39 >= HIDWORD(v39) )
          {
            v29 = v16;
            sub_16CD150((__int64)&v38, v40, 0, 8, v17, v18);
            v19 = (unsigned int)v39;
            v16 = v29;
          }
          v38[v19] = v34;
          LODWORD(v39) = v39 + 1;
          *((_DWORD *)v16 + 3) = a3;
          sub_1E05890((__int64)(v16 + 5), &v33, v19, a3, v17, v18);
        }
        goto LABEL_17;
      }
LABEL_25:
      if ( v14 == (__int64 *)&v37 )
      {
LABEL_5:
        v8 = v39;
        if ( !(_DWORD)v39 )
          break;
      }
      else
      {
        _libc_free((unsigned __int64)v14);
        v8 = v39;
        if ( !(_DWORD)v39 )
          break;
      }
    }
  }
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
  return a3;
}
