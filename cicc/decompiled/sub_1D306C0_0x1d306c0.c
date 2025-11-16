// Function: sub_1D306C0
// Address: 0x1d306c0
//
void __fastcall sub_1D306C0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        int a5,
        unsigned int a6,
        unsigned int a7,
        char a8)
{
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  _QWORD *v13; // r8
  unsigned int v14; // edx
  __int64 v15; // rcx
  __int64 *v16; // r9
  __int64 *v17; // rbx
  __int64 *v18; // r15
  __int64 v19; // r12
  __int64 v20; // r11
  __int64 v21; // r10
  __int64 v22; // rsi
  int v23; // ecx
  __int64 v24; // rax
  __int64 *v25; // rbx
  __int64 *v26; // r12
  __int64 v27; // rsi
  int v28; // ecx
  int v29; // r10d
  int v30; // [rsp+Ch] [rbp-A4h]
  __int64 v31; // [rsp+10h] [rbp-A0h]
  __int64 v32; // [rsp+10h] [rbp-A0h]
  __int64 v33; // [rsp+18h] [rbp-98h]
  _QWORD *v34; // [rsp+18h] [rbp-98h]
  _QWORD *v35; // [rsp+18h] [rbp-98h]
  _QWORD *v36; // [rsp+18h] [rbp-98h]
  unsigned __int64 v40; // [rsp+40h] [rbp-70h] BYREF
  char v41; // [rsp+48h] [rbp-68h]
  char v42; // [rsp+50h] [rbp-60h]
  __int64 *v43; // [rsp+60h] [rbp-50h] BYREF
  __int64 v44; // [rsp+68h] [rbp-48h]
  _BYTE v45[64]; // [rsp+70h] [rbp-40h] BYREF

  if ( a2 != a4 && (*(_BYTE *)(a2 + 26) & 1) != 0 )
  {
    v8 = *(_QWORD *)(a1 + 648);
    v43 = (__int64 *)v45;
    v44 = 0x200000000LL;
    v10 = *(unsigned int *)(v8 + 720);
    if ( (_DWORD)v10 )
    {
      v11 = *(_QWORD *)(v8 + 704);
      LODWORD(v13) = v10 - 1;
      v14 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = v11 + 40LL * v14;
      v16 = *(__int64 **)v15;
      if ( a2 == *(_QWORD *)v15 )
      {
LABEL_6:
        if ( v15 != v11 + 40 * v10 )
        {
          v17 = *(__int64 **)(v15 + 8);
          v18 = &v17[*(unsigned int *)(v15 + 16)];
          if ( v18 != v17 )
          {
            do
            {
              v19 = *v17;
              if ( !*(_DWORD *)(*v17 + 44) && !*(_BYTE *)(v19 + 49) && a3 == *(_DWORD *)(v19 + 8) )
              {
                v20 = *(_QWORD *)(v19 + 16);
                v21 = *(_QWORD *)(v19 + 24);
                if ( !a7
                  || ((v32 = *(_QWORD *)(v19 + 16),
                       v35 = *(_QWORD **)(v19 + 24),
                       sub_15B1350((__int64)&v40, *(unsigned __int64 **)(v21 + 24), *(unsigned __int64 **)(v21 + 32)),
                       !v42)
                   || a6 + a7 <= v40)
                  && (sub_15C4EF0((__int64)&v40, v35, a6, a7), v21 = v40, v20 = v32, v41) )
                {
                  v22 = *(_QWORD *)(v19 + 32);
                  v23 = *(_DWORD *)(v19 + 40);
                  v40 = v22;
                  if ( v22 )
                  {
                    v30 = v23;
                    v31 = v20;
                    v33 = v21;
                    sub_1623A60((__int64)&v40, v22, 2);
                    v23 = v30;
                    v20 = v31;
                    v21 = v33;
                  }
                  v13 = sub_1D24380(a1, v20, v21, a4, a5, *(_BYTE *)(v19 + 48), (__int64 *)&v40, v23);
                  if ( v40 )
                  {
                    v34 = v13;
                    sub_161E7C0((__int64)&v40, v40);
                    v13 = v34;
                  }
                  v24 = (unsigned int)v44;
                  if ( (unsigned int)v44 >= HIDWORD(v44) )
                  {
                    v36 = v13;
                    sub_16CD150((__int64)&v43, v45, 0, 8, (int)v13, (int)v16);
                    v24 = (unsigned int)v44;
                    v13 = v36;
                  }
                  v43[v24] = (__int64)v13;
                  LODWORD(v44) = v44 + 1;
                  if ( a8 )
                    *(_BYTE *)(v19 + 49) = 1;
                }
              }
              ++v17;
            }
            while ( v18 != v17 );
            v25 = v43;
            v26 = &v43[(unsigned int)v44];
            if ( v43 != v26 )
            {
              do
              {
                v27 = *v25++;
                sub_1D30360(a1, v27, a4, 0, (int)v13, v16);
              }
              while ( v26 != v25 );
              v26 = v43;
            }
            if ( v26 != (__int64 *)v45 )
              _libc_free((unsigned __int64)v26);
          }
        }
      }
      else
      {
        v28 = 1;
        while ( v16 != (__int64 *)-8LL )
        {
          v29 = v28 + 1;
          v14 = (unsigned int)v13 & (v28 + v14);
          v15 = v11 + 40LL * v14;
          v16 = *(__int64 **)v15;
          if ( a2 == *(_QWORD *)v15 )
            goto LABEL_6;
          v28 = v29;
        }
      }
    }
  }
}
