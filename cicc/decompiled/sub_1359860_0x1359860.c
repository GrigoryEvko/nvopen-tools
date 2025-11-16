// Function: sub_1359860
// Address: 0x1359860
//
void __fastcall sub_1359860(__int64 a1, __int64 a2)
{
  int v2; // r12d
  __int64 v3; // r14
  int v6; // edx
  unsigned int v7; // eax
  _QWORD *v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rcx
  char v11; // r13
  _QWORD *v12; // r13
  __int64 v13; // r14
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdi
  int v20; // edx
  int v21; // esi
  __int64 v22; // rdi
  int v23; // eax
  int v24; // ecx
  int v25; // eax
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rax
  int v29; // eax
  int v30; // edx
  __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // r8
  int v34; // r9d
  __int64 v35; // [rsp+0h] [rbp-A0h]
  __int64 v36; // [rsp+8h] [rbp-98h]
  __int64 v37; // [rsp+8h] [rbp-98h]
  void *v38; // [rsp+10h] [rbp-90h] BYREF
  char v39[16]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v40; // [rsp+28h] [rbp-78h]
  void *v41; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v42[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v43; // [rsp+58h] [rbp-48h]
  __int64 v44; // [rsp+60h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 48);
  if ( v2 )
  {
    v3 = *(_QWORD *)(a1 + 32);
    sub_1359800(&v38, -8, 0);
    sub_1359800(&v41, -16, 0);
    v6 = v2 - 1;
    v7 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (_QWORD *)(v3 + 48LL * v7);
    v9 = v8[3];
    if ( a2 == v9 )
    {
      v10 = v43;
      v11 = 1;
    }
    else
    {
      v10 = v43;
      v33 = (_QWORD *)(v3 + 48LL * v7);
      v34 = 1;
      v8 = 0;
      while ( v9 != v40 )
      {
        if ( v9 != v43 || v8 )
          v33 = v8;
        v7 = v6 & (v34 + v7);
        v8 = (_QWORD *)(v3 + 48LL * v7);
        v9 = v8[3];
        if ( v9 == a2 )
        {
          v11 = 1;
          goto LABEL_5;
        }
        ++v34;
        v8 = v33;
        v33 = (_QWORD *)(v3 + 48LL * v7);
      }
      v11 = 0;
      if ( !v8 )
        v8 = v33;
    }
LABEL_5:
    v41 = &unk_49EE2B0;
    if ( v10 != 0 && v10 != -8 && v10 != -16 )
      sub_1649B30(v42);
    v38 = &unk_49EE2B0;
    if ( v40 != -8 && v40 != 0 && v40 != -16 )
      sub_1649B30(v39);
    if ( v11 && v8 != (_QWORD *)(*(_QWORD *)(a1 + 32) + 48LL * *(unsigned int *)(a1 + 48)) )
    {
      v12 = (_QWORD *)v8[5];
      v13 = v12[3];
      v14 = *(_QWORD *)(v13 + 32);
      if ( v14 )
      {
        v15 = *(_QWORD *)(v14 + 32);
        v36 = *(_QWORD *)(v13 + 32);
        v16 = v36;
        if ( v15 )
        {
          v17 = sub_1357F10(v15, a1);
          v18 = v36;
          v16 = *(_QWORD *)(v36 + 32);
          if ( v17 != v16 )
          {
            *(_DWORD *)(v17 + 64) = (*(_DWORD *)(v17 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v17 + 64) & 0xF8000000;
            v19 = *(_QWORD *)(v36 + 32);
            v20 = *(_DWORD *)(v19 + 64);
            v21 = (v20 + 0x7FFFFFF) & 0x7FFFFFF;
            *(_DWORD *)(v19 + 64) = v21 | v20 & 0xF8000000;
            if ( !v21 )
            {
              v35 = v17;
              sub_1357730(v19, a1);
              v17 = v35;
              v18 = v36;
            }
            *(_QWORD *)(v18 + 32) = v17;
            v16 = v17;
          }
          if ( *(_QWORD *)(v13 + 32) != v16 )
          {
            *(_DWORD *)(v16 + 64) = (*(_DWORD *)(v16 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v16 + 64) & 0xF8000000;
            v22 = *(_QWORD *)(v13 + 32);
            v23 = *(_DWORD *)(v22 + 64);
            v24 = (v23 + 0x7FFFFFF) & 0x7FFFFFF;
            *(_DWORD *)(v22 + 64) = v24 | v23 & 0xF8000000;
            if ( !v24 )
            {
              v37 = v16;
              sub_1357730(v22, a1);
              v16 = v37;
            }
            *(_QWORD *)(v13 + 32) = v16;
          }
        }
        v12[3] = v16;
        *(_DWORD *)(v16 + 64) = (*(_DWORD *)(v16 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v16 + 64) & 0xF8000000;
        v25 = *(_DWORD *)(v13 + 64);
        v26 = (v25 + 0x7FFFFFF) & 0x7FFFFFF;
        *(_DWORD *)(v13 + 64) = v26 | v25 & 0xF8000000;
        if ( !v26 )
          sub_1357730(v13, a1);
        v13 = v12[3];
      }
      v27 = v12[2];
      if ( v27 )
      {
        *(_QWORD *)(v27 + 8) = v12[1];
        v27 = v12[2];
      }
      *(_QWORD *)v12[1] = v27;
      v28 = v12[3];
      if ( *(_QWORD **)(v28 + 24) == v12 + 2 )
        *(_QWORD *)(v28 + 24) = v12[1];
      j_j___libc_free_0(v12, 64);
      if ( (*(_BYTE *)(v13 + 67) & 0x40) != 0 )
      {
        --*(_DWORD *)(v13 + 68);
        --*(_DWORD *)(a1 + 56);
      }
      v29 = *(_DWORD *)(v13 + 64);
      v30 = (v29 + 0x7FFFFFF) & 0x7FFFFFF;
      *(_DWORD *)(v13 + 64) = v30 | v29 & 0xF8000000;
      if ( !v30 )
        sub_1357730(v13, a1);
      sub_1359800(&v41, -16, 0);
      v31 = v8[3];
      v32 = v43;
      if ( v31 != v43 )
      {
        if ( v31 != -8 && v31 != 0 && v31 != -16 )
        {
          sub_1649B30(v8 + 1);
          v32 = v43;
        }
        v8[3] = v32;
        if ( v32 != 0 && v32 != -8 && v32 != -16 )
          sub_1649AC0(v8 + 1, v42[0] & 0xFFFFFFFFFFFFFFF8LL);
        v32 = v43;
      }
      v8[4] = v44;
      v41 = &unk_49EE2B0;
      if ( v32 != -8 && v32 != 0 && v32 != -16 )
        sub_1649B30(v42);
      --*(_DWORD *)(a1 + 40);
      ++*(_DWORD *)(a1 + 44);
    }
  }
}
