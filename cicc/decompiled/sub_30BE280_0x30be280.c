// Function: sub_30BE280
// Address: 0x30be280
//
__int64 *__fastcall sub_30BE280(__int64 a1)
{
  __int64 v2; // rdx
  __int64 *result; // rax
  __int64 v4; // r12
  __int64 v5; // r9
  _BYTE *v6; // rdi
  _BYTE *v7; // r13
  char v8; // r10
  __int64 v9; // rbx
  __int64 *v10; // rdx
  int v11; // esi
  __int64 v12; // rdi
  int v13; // esi
  __int64 v14; // rcx
  __int64 *v15; // rax
  __int64 v16; // r8
  __int64 v17; // r14
  __int64 *v18; // rax
  char v19; // dl
  __int64 (__fastcall *v20)(__int64, __int64, __int64); // rax
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // eax
  __int64 *v24; // [rsp+18h] [rbp-D8h]
  __int64 *v25; // [rsp+28h] [rbp-C8h]
  _BYTE *v26; // [rsp+30h] [rbp-C0h]
  __int64 v27; // [rsp+58h] [rbp-98h] BYREF
  _BYTE *v28; // [rsp+60h] [rbp-90h] BYREF
  __int64 v29; // [rsp+68h] [rbp-88h]
  _BYTE v30[16]; // [rsp+70h] [rbp-80h] BYREF
  __int64 (*v31)(); // [rsp+80h] [rbp-70h] BYREF
  __int64 *v32; // [rsp+88h] [rbp-68h]
  __int64 v33; // [rsp+90h] [rbp-60h]
  int v34; // [rsp+98h] [rbp-58h]
  char v35; // [rsp+9Ch] [rbp-54h]
  char v36; // [rsp+A0h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  result = *(__int64 **)(v2 + 96);
  v24 = &result[*(unsigned int *)(v2 + 104)];
  if ( result != v24 )
  {
    v25 = *(__int64 **)(v2 + 96);
    do
    {
      v4 = *v25;
      v28 = v30;
      v29 = 0x200000000LL;
      v31 = sub_30B9550;
      v32 = &v27;
      sub_30B0A30(v4, (__int64)&v31, (__int64)&v28);
      v6 = v28;
      v31 = 0;
      v33 = 4;
      v32 = (__int64 *)&v36;
      v34 = 0;
      v35 = 1;
      v26 = &v28[8 * (unsigned int)v29];
      if ( v26 != v28 )
      {
        v7 = v28;
        v8 = 1;
        do
        {
          v9 = *(_QWORD *)(*(_QWORD *)v7 + 16LL);
          if ( v9 )
          {
            while ( 1 )
            {
              while ( 1 )
              {
                v10 = *(__int64 **)(v9 + 24);
                if ( *(_BYTE *)v10 > 0x1Cu )
                {
                  v11 = *(_DWORD *)(a1 + 56);
                  v12 = *(_QWORD *)(a1 + 40);
                  if ( v11 )
                  {
                    v13 = v11 - 1;
                    v14 = v13 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
                    v15 = (__int64 *)(v12 + 16 * v14);
                    v16 = *v15;
                    if ( v10 != (__int64 *)*v15 )
                    {
                      v23 = 1;
                      while ( v16 != -4096 )
                      {
                        v5 = (unsigned int)(v23 + 1);
                        v14 = v13 & (unsigned int)(v23 + v14);
                        v15 = (__int64 *)(v12 + 16LL * (unsigned int)v14);
                        v16 = *v15;
                        if ( v10 == (__int64 *)*v15 )
                          goto LABEL_9;
                        v23 = v5;
                      }
                      goto LABEL_16;
                    }
LABEL_9:
                    v17 = v15[1];
                    if ( v4 != v17 )
                    {
                      if ( v17 )
                        break;
                    }
                  }
                }
LABEL_16:
                v9 = *(_QWORD *)(v9 + 8);
                if ( !v9 )
                  goto LABEL_17;
              }
              if ( !v8 )
                goto LABEL_25;
              v18 = v32;
              v14 = HIDWORD(v33);
              v10 = &v32[HIDWORD(v33)];
              if ( v32 != v10 )
              {
                while ( v17 != *v18 )
                {
                  if ( v10 == ++v18 )
                    goto LABEL_31;
                }
                goto LABEL_16;
              }
LABEL_31:
              if ( HIDWORD(v33) < (unsigned int)v33 )
              {
                ++HIDWORD(v33);
                *v10 = v17;
                v22 = *(_QWORD *)a1;
                v31 = (__int64 (*)())((char *)v31 + 1);
                v20 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(v22 + 40);
                if ( v20 != sub_30B3030 )
                  goto LABEL_33;
LABEL_27:
                v21 = sub_22077B0(0x10u);
                if ( v21 )
                {
                  *(_QWORD *)v21 = v17;
                  *(_DWORD *)(v21 + 8) = 1;
                }
                v27 = v21;
                sub_30B2AC0(v4 + 8, &v27);
                v9 = *(_QWORD *)(v9 + 8);
                v8 = v35;
                if ( !v9 )
                  break;
              }
              else
              {
LABEL_25:
                sub_C8CC70((__int64)&v31, v17, (__int64)v10, v14, v16, v5);
                v8 = v35;
                if ( !v19 )
                  goto LABEL_16;
                v20 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 40LL);
                if ( v20 == sub_30B3030 )
                  goto LABEL_27;
LABEL_33:
                v20(a1, v4, v17);
                v9 = *(_QWORD *)(v9 + 8);
                v8 = v35;
                if ( !v9 )
                  break;
              }
            }
          }
LABEL_17:
          v7 += 8;
        }
        while ( v26 != v7 );
        if ( !v8 )
          _libc_free((unsigned __int64)v32);
        v6 = v28;
      }
      if ( v6 != v30 )
        _libc_free((unsigned __int64)v6);
      result = ++v25;
    }
    while ( v24 != v25 );
  }
  return result;
}
