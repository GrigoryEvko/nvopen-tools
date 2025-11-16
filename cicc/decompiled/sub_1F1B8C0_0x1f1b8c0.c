// Function: sub_1F1B8C0
// Address: 0x1f1b8c0
//
void __fastcall sub_1F1B8C0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r14
  unsigned int v7; // eax
  int v8; // r15d
  __int64 v9; // r12
  int v10; // ebx
  int v11; // esi
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // r15
  __int64 *v17; // rdx
  int v18; // r8d
  int v19; // r9d
  __int64 *v20; // rax
  char v21; // dl
  __int64 v22; // rax
  unsigned __int64 v23; // r12
  int v24; // eax
  __int64 *v25; // rsi
  __int64 *v26; // rcx
  __int64 v27; // rax
  __int64 v28; // [rsp+10h] [rbp-E0h]
  __int64 v29; // [rsp+18h] [rbp-D8h]
  _QWORD *v30; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v31; // [rsp+28h] [rbp-C8h]
  _QWORD v32[4]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+50h] [rbp-A0h] BYREF
  __int64 *v34; // [rsp+58h] [rbp-98h]
  __int64 *v35; // [rsp+60h] [rbp-90h]
  __int64 v36; // [rsp+68h] [rbp-88h]
  int v37; // [rsp+70h] [rbp-80h]
  _QWORD v38[15]; // [rsp+78h] [rbp-78h] BYREF

  v3 = v32;
  v4 = *(_QWORD *)(a1 + 72);
  v34 = v38;
  v35 = v38;
  v36 = 0x100000008LL;
  v31 = 0x400000001LL;
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL);
  v30 = v32;
  v33 = 1;
  v37 = 0;
  v38[0] = a2;
  v32[0] = a2;
  v6 = *(_QWORD *)(v4 + 8);
  v29 = v5;
  v7 = 1;
  while ( 1 )
  {
    v8 = 0;
    v9 = v3[v7 - 1];
    LODWORD(v31) = v7 - 1;
    v10 = *(_DWORD *)(*(_QWORD *)(v4 + 16) + 8LL) - *(_DWORD *)(v4 + 64);
    if ( v10 )
    {
      do
      {
        v11 = v8++;
        sub_1F1B3E0(a1, v11, (int *)v9);
      }
      while ( v10 != v8 );
    }
    v12 = *(_QWORD *)(v9 + 8);
    if ( (v12 & 6) == 0 )
    {
      v13 = sub_1DA9310(v29, v12);
      v28 = *(_QWORD *)(v13 + 72);
      if ( v28 != *(_QWORD *)(v13 + 64) )
      {
        v14 = *(_QWORD *)(v13 + 64);
        while ( 1 )
        {
          v22 = *(_QWORD *)(*(_QWORD *)(v29 + 392) + 16LL * *(unsigned int *)(*(_QWORD *)v14 + 48LL) + 8);
          v23 = v22 & 0xFFFFFFFFFFFFFFF8LL;
          v24 = (v22 >> 1) & 3;
          if ( v24 )
            v15 = (2LL * (v24 - 1)) | v23;
          else
            v15 = *(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL | 6;
          v16 = 0;
          v17 = (__int64 *)sub_1DB3C70((__int64 *)v6, v15);
          if ( v17 != (__int64 *)(*(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8))
            && (*(_DWORD *)((*v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v17 >> 1) & 3) <= (*(_DWORD *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v15 >> 1) & 3) )
          {
            v16 = v17[2];
          }
          v20 = v34;
          if ( v35 == v34 )
          {
            v25 = &v34[HIDWORD(v36)];
            if ( v34 != v25 )
            {
              v26 = 0;
              while ( v16 != *v20 )
              {
                if ( *v20 == -2 )
                  v26 = v20;
                if ( v25 == ++v20 )
                {
                  if ( !v26 )
                    goto LABEL_28;
                  *v26 = v16;
                  --v37;
                  ++v33;
                  goto LABEL_26;
                }
              }
              goto LABEL_15;
            }
LABEL_28:
            if ( HIDWORD(v36) < (unsigned int)v36 )
              break;
          }
          sub_16CCBA0((__int64)&v33, v16);
          if ( v21 )
          {
LABEL_26:
            v27 = (unsigned int)v31;
            if ( (unsigned int)v31 < HIDWORD(v31) )
            {
LABEL_27:
              v30[v27] = v16;
              LODWORD(v31) = v31 + 1;
              goto LABEL_15;
            }
LABEL_30:
            sub_16CD150((__int64)&v30, v32, 0, 8, v18, v19);
            v27 = (unsigned int)v31;
            goto LABEL_27;
          }
LABEL_15:
          v14 += 8;
          if ( v28 == v14 )
            goto LABEL_5;
        }
        ++HIDWORD(v36);
        *v25 = v16;
        v27 = (unsigned int)v31;
        ++v33;
        if ( (unsigned int)v31 < HIDWORD(v31) )
          goto LABEL_27;
        goto LABEL_30;
      }
    }
LABEL_5:
    v7 = v31;
    v3 = v30;
    if ( !(_DWORD)v31 )
      break;
    v4 = *(_QWORD *)(a1 + 72);
  }
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  if ( v35 != v34 )
    _libc_free((unsigned __int64)v35);
}
