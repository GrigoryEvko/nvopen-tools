// Function: sub_1BA9FD0
// Address: 0x1ba9fd0
//
__int64 __fastcall sub_1BA9FD0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r15
  __int64 v5; // rdx
  int v6; // r12d
  unsigned int v7; // r12d
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rax
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rdx
  unsigned __int64 v14; // r13
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rbx
  const void *v21; // r8
  __int64 *v22; // r9
  __int64 v23; // rdx
  __int64 *v24; // r13
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // r12
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // [rsp+8h] [rbp-68h]
  __int64 *v32; // [rsp+8h] [rbp-68h]
  __int64 *v33; // [rsp+8h] [rbp-68h]
  const void *v34; // [rsp+10h] [rbp-60h]
  const void *v35; // [rsp+10h] [rbp-60h]
  _BYTE *v36; // [rsp+20h] [rbp-50h] BYREF
  __int64 v37; // [rsp+28h] [rbp-48h]
  _BYTE v38[64]; // [rsp+30h] [rbp-40h] BYREF

  v3 = 0;
  if ( *(_BYTE *)(a2 + 16) == 77 )
  {
    v5 = *(_QWORD *)(a2 + 40);
    if ( v5 != **(_QWORD **)(*(_QWORD *)a1 + 32LL) )
    {
      v6 = *(_DWORD *)(a2 + 20);
      v36 = v38;
      v37 = 0x200000000LL;
      v7 = v6 & 0xFFFFFFF;
      if ( v7 )
      {
        v8 = 8LL * v7;
        while ( 1 )
        {
          if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
            v9 = *(_QWORD *)(a2 - 8);
          else
            v9 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v10 = sub_1BA99C0(a1, *(_QWORD *)(v3 + v9 + 24LL * *(unsigned int *)(a2 + 56) + 8), v5, a3);
          if ( v10 )
          {
            v13 = (unsigned int)v37;
            if ( (unsigned int)v37 >= HIDWORD(v37) )
            {
              v31 = v10;
              sub_16CD150((__int64)&v36, v38, 0, 8, v11, v12);
              v13 = (unsigned int)v37;
              v10 = v31;
            }
            *(_QWORD *)&v36[8 * v13] = v10;
            LODWORD(v37) = v37 + 1;
          }
          v3 += 8;
          if ( v8 == v3 )
            break;
          v5 = *(_QWORD *)(a2 + 40);
        }
        v14 = (unsigned __int64)v36;
        v15 = (unsigned int)v37;
        v16 = sub_22077B0(56);
        v3 = v16;
        if ( v16 )
        {
          *(_QWORD *)(v16 + 8) = 0;
          *(_QWORD *)(v16 + 16) = 0;
          *(_BYTE *)(v16 + 24) = 0;
          *(_QWORD *)(v16 + 32) = 0;
          *(_QWORD *)(v16 + 40) = a2;
          *(_QWORD *)(v16 + 48) = 0;
          *(_QWORD *)v16 = &unk_49F6F48;
          if ( v15 )
          {
            v19 = sub_22077B0(72);
            v20 = v19;
            if ( v19 )
            {
              *(_BYTE *)v19 = 1;
              v21 = (const void *)(v19 + 56);
              *(_QWORD *)(v19 + 8) = v19 + 24;
              v22 = (__int64 *)(v14 + 8 * v15);
              v23 = v19 + 56;
              v24 = (__int64 *)(v14 + 8);
              *(_QWORD *)(v19 + 32) = 0;
              *(_QWORD *)(v19 + 16) = 0x100000000LL;
              *(_QWORD *)(v19 + 48) = 0x200000000LL;
              v25 = 0;
              *(_QWORD *)(v20 + 40) = v20 + 56;
              v26 = *(v24 - 1);
              while ( 1 )
              {
                *(_QWORD *)(v23 + 8 * v25) = v26;
                ++*(_DWORD *)(v20 + 48);
                v27 = *(unsigned int *)(v26 + 16);
                if ( (unsigned int)v27 >= *(_DWORD *)(v26 + 20) )
                {
                  v33 = v22;
                  v35 = v21;
                  sub_16CD150(v26 + 8, (const void *)(v26 + 24), 0, 8, (int)v21, (int)v22);
                  v27 = *(unsigned int *)(v26 + 16);
                  v22 = v33;
                  v21 = v35;
                }
                *(_QWORD *)(*(_QWORD *)(v26 + 8) + 8 * v27) = v20;
                ++*(_DWORD *)(v26 + 16);
                if ( v22 == v24 )
                  break;
                v26 = *v24;
                v25 = *(unsigned int *)(v20 + 48);
                if ( (unsigned int)v25 >= *(_DWORD *)(v20 + 52) )
                {
                  v32 = v22;
                  v34 = v21;
                  sub_16CD150(v20 + 40, v21, 0, 8, (int)v21, (int)v22);
                  v25 = *(unsigned int *)(v20 + 48);
                  v22 = v32;
                  v21 = v34;
                }
                v23 = *(_QWORD *)(v20 + 40);
                ++v24;
              }
            }
            v28 = *(_QWORD *)(v3 + 48);
            *(_QWORD *)(v3 + 48) = v20;
            if ( v28 )
            {
              v29 = *(_QWORD *)(v28 + 40);
              if ( v29 != v28 + 56 )
                _libc_free(v29);
              v30 = *(_QWORD *)(v28 + 8);
              if ( v30 != v28 + 24 )
                _libc_free(v30);
              j_j___libc_free_0(v28, 72);
            }
          }
        }
      }
      else
      {
        v18 = sub_22077B0(56);
        v3 = v18;
        if ( v18 )
        {
          *(_QWORD *)(v18 + 8) = 0;
          *(_QWORD *)(v18 + 16) = 0;
          *(_BYTE *)(v18 + 24) = 0;
          *(_QWORD *)(v18 + 32) = 0;
          *(_QWORD *)v18 = &unk_49F6F48;
          *(_QWORD *)(v18 + 40) = a2;
          *(_QWORD *)(v18 + 48) = 0;
        }
      }
      if ( v36 != v38 )
        _libc_free((unsigned __int64)v36);
    }
  }
  return v3;
}
