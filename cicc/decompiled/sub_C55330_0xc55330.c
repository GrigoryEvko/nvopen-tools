// Function: sub_C55330
// Address: 0xc55330
//
__int64 __fastcall sub_C55330(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rbx
  bool v13; // zf
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 v18; // rbx
  unsigned __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 i; // rbx
  void (__fastcall *v23)(__int64, __int64, __int64); // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r12
  __int64 v27; // rdi
  _DWORD *v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 v33; // r12
  void (__fastcall *v34)(__int64, __int64, __int64); // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // [rsp-8h] [rbp-60h]
  __int64 v38; // [rsp+0h] [rbp-58h]
  _BYTE v39[49]; // [rsp+27h] [rbp-31h] BYREF

  v39[0] = 0;
  result = sub_C54F80(a1 + 152, a1, a3, a4, a5, a6, v39);
  v10 = v37;
  v11 = v38;
  if ( !(_BYTE)result )
  {
    if ( v39[0] )
    {
      if ( !*(_QWORD *)(sub_C4F990(v38) + 1456) )
      {
        v14 = sub_C4F990(v38);
        v16 = *(_QWORD *)(v14 + 1480);
        v17 = *(_QWORD *)(v14 + 1472);
        v18 = v14;
        v19 = v16 - v17;
        if ( v16 == v17 )
        {
          v21 = 0;
        }
        else
        {
          if ( v19 > 0x7FFFFFFFFFFFFFE0LL )
            sub_4261EA(v38, v37, v15);
          v11 = *(_QWORD *)(v14 + 1480) - v17;
          v20 = sub_22077B0(v19);
          v16 = *(_QWORD *)(v18 + 1480);
          v17 = *(_QWORD *)(v18 + 1472);
          v21 = v20;
        }
        for ( i = v21; v16 != v17; i += 32 )
        {
          if ( i )
          {
            *(_QWORD *)(i + 16) = 0;
            v23 = *(void (__fastcall **)(__int64, __int64, __int64))(v17 + 16);
            if ( v23 )
            {
              v11 = i;
              v23(i, v17, 2);
              *(_QWORD *)(i + 24) = *(_QWORD *)(v17 + 24);
              *(_QWORD *)(i + 16) = *(_QWORD *)(v17 + 16);
            }
          }
          v17 += 32;
        }
        v24 = sub_CB7210(v11);
        v25 = *(_QWORD *)(v24 + 32);
        v26 = v24;
        if ( (unsigned __int64)(*(_QWORD *)(v24 + 24) - v25) <= 5 )
        {
          v27 = sub_CB6200(v24, "NVIDIA", 6);
        }
        else
        {
          *(_DWORD *)v25 = 1145656910;
          v27 = v24;
          *(_WORD *)(v25 + 4) = 16713;
          *(_QWORD *)(v24 + 32) += 6LL;
        }
        sub_904010(v27, " ");
        v28 = *(_DWORD **)(v26 + 32);
        if ( *(_QWORD *)(v26 + 24) - (_QWORD)v28 <= 3u )
        {
          v36 = sub_CB6200(v26, "NVVM", 4);
          v30 = *(_QWORD *)(v36 + 32);
          v29 = v36;
        }
        else
        {
          *v28 = 1297503822;
          v29 = v26;
          v30 = *(_QWORD *)(v26 + 32) + 4LL;
          *(_QWORD *)(v26 + 32) = v30;
        }
        if ( (unsigned __int64)(*(_QWORD *)(v29 + 24) - v30) <= 8 )
        {
          v35 = sub_CB6200(v29, " version ", 9);
          v31 = *(_QWORD *)(v35 + 32);
          v29 = v35;
        }
        else
        {
          *(_BYTE *)(v30 + 8) = 32;
          *(_QWORD *)v30 = 0x6E6F697372657620LL;
          v31 = *(_QWORD *)(v29 + 32) + 9LL;
          *(_QWORD *)(v29 + 32) = v31;
        }
        if ( (unsigned __int64)(*(_QWORD *)(v29 + 24) - v31) <= 5 )
        {
          v29 = sub_CB6200(v29, "20.0.0", 6);
        }
        else
        {
          *(_DWORD *)v31 = 808333362;
          *(_WORD *)(v31 + 4) = 12334;
          *(_QWORD *)(v29 + 32) += 6LL;
        }
        sub_904010(v29, "\n  ");
        sub_904010(v26, "Optimized build");
        v11 = v26;
        v32 = v21;
        sub_904010(v11, ".\n");
        if ( v21 != i )
        {
          do
          {
            v10 = sub_CB7210(v11);
            if ( !*(_QWORD *)(v32 + 16) )
              goto LABEL_39;
            v11 = v32;
            (*(void (__fastcall **)(__int64, __int64))(v32 + 24))(v32, v10);
            v32 += 32;
          }
          while ( v32 != i );
          v33 = v21;
          do
          {
            v34 = *(void (__fastcall **)(__int64, __int64, __int64))(v33 + 16);
            if ( v34 )
              v34(v33, v33, 3);
            v33 += 32;
          }
          while ( v33 != i );
        }
        if ( v21 )
          j_j___libc_free_0(v21, v19);
        exit(0);
      }
      v12 = sub_C4F990(v38);
      v10 = sub_CB7210(v38);
      if ( *(_QWORD *)(v12 + 1456) )
      {
        (*(void (__fastcall **)(__int64, __int64))(v12 + 1464))(v12 + 1440, v10);
        exit(0);
      }
LABEL_39:
      sub_4263D6(v11, v10, v9);
    }
    v13 = *(_QWORD *)(a1 + 176) == 0;
    *(_WORD *)(a1 + 14) = a2;
    if ( v13 )
      goto LABEL_39;
    (*(void (__fastcall **)(__int64, _BYTE *))(a1 + 184))(a1 + 160, v39);
    return 0;
  }
  return result;
}
