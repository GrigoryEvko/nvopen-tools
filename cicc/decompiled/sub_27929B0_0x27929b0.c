// Function: sub_27929B0
// Address: 0x27929b0
//
void __fastcall sub_27929B0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // edi
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // r12
  __int64 v14; // rdx
  int v15; // ecx
  __int64 v16; // r15
  _DWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 v24; // rbx
  int v25; // edx
  int v26; // esi
  __int64 v27; // [rsp+8h] [rbp-D8h]
  __int64 v28; // [rsp+10h] [rbp-D0h]
  _DWORD *v29; // [rsp+18h] [rbp-C8h]
  __int64 v30; // [rsp+18h] [rbp-C8h]
  _DWORD *v31; // [rsp+28h] [rbp-B8h] BYREF
  int v32; // [rsp+30h] [rbp-B0h]
  char v33; // [rsp+34h] [rbp-ACh]
  __int64 v34; // [rsp+38h] [rbp-A8h]
  char *v35; // [rsp+40h] [rbp-A0h]
  __int64 v36; // [rsp+48h] [rbp-98h]
  char v37; // [rsp+50h] [rbp-90h] BYREF
  __int64 v38; // [rsp+60h] [rbp-80h]
  int v39; // [rsp+70h] [rbp-70h]
  char v40; // [rsp+74h] [rbp-6Ch]
  __int64 v41; // [rsp+78h] [rbp-68h]
  _BYTE *v42; // [rsp+80h] [rbp-60h] BYREF
  __int64 v43; // [rsp+88h] [rbp-58h]
  _BYTE v44[16]; // [rsp+90h] [rbp-50h] BYREF
  __int64 v45; // [rsp+A0h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  v7 = sub_C7D670((unsigned __int64)v6 << 6, 8);
  *(_QWORD *)(a1 + 8) = v7;
  if ( v5 )
  {
    v11 = *(unsigned int *)(a1 + 24);
    v12 = v4 << 6;
    v40 = 0;
    v42 = v44;
    v13 = v5 + v12;
    v43 = 0x400000000LL;
    v14 = v7 + (v11 << 6);
    *(_QWORD *)(a1 + 16) = 0;
    v39 = -1;
    v41 = 0;
    v45 = 0;
    if ( v7 != v14 )
    {
      do
      {
        if ( v7 )
        {
          v15 = v39;
          *(_DWORD *)(v7 + 24) = 0;
          *(_DWORD *)(v7 + 28) = 4;
          *(_DWORD *)v7 = v15;
          *(_BYTE *)(v7 + 4) = v40;
          *(_QWORD *)(v7 + 8) = v41;
          *(_QWORD *)(v7 + 16) = v7 + 32;
          if ( (_DWORD)v43 )
          {
            v27 = v14;
            v28 = v7;
            sub_2789770(v7 + 16, (__int64)&v42, v14, (unsigned int)v43, v9, v10);
            v14 = v27;
            v7 = v28;
          }
          *(_QWORD *)(v7 + 48) = v45;
        }
        v7 += 64;
      }
      while ( v14 != v7 );
      if ( v42 != v44 )
        _libc_free((unsigned __int64)v42);
    }
    v40 = 0;
    v35 = &v37;
    v32 = -1;
    v33 = 0;
    v34 = 0;
    v36 = 0x400000000LL;
    v38 = 0;
    v39 = -2;
    v41 = 0;
    v42 = v44;
    v43 = 0x400000000LL;
    v45 = 0;
    if ( v13 != v5 )
    {
      v16 = v5;
      do
      {
        if ( *(_DWORD *)v16 <= 0xFFFFFFFD )
        {
          sub_278F8C0(a1, v16, (__int64 *)&v31);
          v17 = v31;
          *v31 = *(_DWORD *)v16;
          v29 = v17;
          *((_BYTE *)v17 + 4) = *(_BYTE *)(v16 + 4);
          v18 = *(_QWORD *)(v16 + 8);
          *((_QWORD *)v17 + 1) = v18;
          sub_2789850((__int64)(v17 + 4), (char **)(v16 + 16), v18, v19, v20, v21);
          *((_QWORD *)v29 + 6) = *(_QWORD *)(v16 + 48);
          v31[14] = *(_DWORD *)(v16 + 56);
          ++*(_DWORD *)(a1 + 16);
        }
        v22 = *(_QWORD *)(v16 + 16);
        if ( v22 != v16 + 32 )
          _libc_free(v22);
        v16 += 64;
      }
      while ( v13 != v16 );
    }
    sub_C7D6A0(v5, v12, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v23 = *(unsigned int *)(a1 + 24);
    v39 = -1;
    v40 = 0;
    v24 = v7 + (v23 << 6);
    v41 = 0;
    v42 = v44;
    v43 = 0x400000000LL;
    v45 = 0;
    if ( v7 != v24 )
    {
      do
      {
        if ( v7 )
        {
          v25 = v39;
          v26 = v43;
          *(_DWORD *)(v7 + 24) = 0;
          *(_DWORD *)(v7 + 28) = 4;
          *(_DWORD *)v7 = v25;
          *(_BYTE *)(v7 + 4) = v40;
          *(_QWORD *)(v7 + 8) = v41;
          *(_QWORD *)(v7 + 16) = v7 + 32;
          if ( v26 )
          {
            v30 = v7;
            sub_2789770(v7 + 16, (__int64)&v42, v7 + 32, v8, v9, v10);
            v7 = v30;
          }
          *(_QWORD *)(v7 + 48) = v45;
        }
        v7 += 64;
      }
      while ( v24 != v7 );
      if ( v42 != v44 )
        _libc_free((unsigned __int64)v42);
    }
  }
}
