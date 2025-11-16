// Function: sub_31DE320
// Address: 0x31de320
//
void __fastcall sub_31DE320(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rbx
  unsigned int **v11; // r11
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned int v14; // r13d
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned int *v17; // rdx
  __int64 v18; // rcx
  unsigned int *v19; // rdi
  __int64 v20; // rbx
  unsigned int *v21; // rdx
  __int64 v22; // rax
  unsigned int *v23; // rdx
  unsigned int v24; // ebx
  __int64 v25; // rcx
  unsigned int v26; // [rsp+0h] [rbp-D0h]
  unsigned int **v27; // [rsp+0h] [rbp-D0h]
  __int64 v28; // [rsp+8h] [rbp-C8h]
  unsigned int v29; // [rsp+8h] [rbp-C8h]
  char v30; // [rsp+10h] [rbp-C0h]
  unsigned int **v31; // [rsp+10h] [rbp-C0h]
  __int64 v32; // [rsp+10h] [rbp-C0h]
  char v33; // [rsp+18h] [rbp-B8h]
  unsigned int *v34; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v35; // [rsp+28h] [rbp-A8h]
  _BYTE v36[48]; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int *v37; // [rsp+60h] [rbp-70h] BYREF
  __int64 v38; // [rsp+68h] [rbp-68h]
  _BYTE v39[96]; // [rsp+70h] [rbp-60h] BYREF

  v1 = *(__int64 **)(a1 + 232);
  v2 = v1[8];
  if ( !v2 || *(_DWORD *)v2 == 5 || *(_QWORD *)(v2 + 8) == *(_QWORD *)(v2 + 16) )
    return;
  v3 = *v1;
  v4 = sub_31DA6B0(a1);
  v5 = (*(unsigned __int8 (__fastcall **)(__int64, bool, __int64))(*(_QWORD *)v4 + 120LL))(
         v4,
         (unsigned int)(*(_DWORD *)v2 - 3) <= 1,
         v3)
     ^ 1u;
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 200) + 879LL) & 2) != 0 )
  {
    v34 = (unsigned int *)v36;
    v37 = (unsigned int *)v39;
    v38 = 0xC00000000LL;
    v7 = *(_QWORD *)(v2 + 16);
    v35 = 0xC00000000LL;
    v8 = *(_QWORD *)(v2 + 8);
    v9 = (v7 - v8) >> 5;
    if ( (_DWORD)v9 )
    {
      v9 = (unsigned int)v9;
      v10 = 0;
      v11 = &v34;
      while ( 1 )
      {
        v14 = v10;
        if ( *(_DWORD *)(v8 + 32 * v10 + 24) == 1 )
        {
          v15 = (unsigned int)v38;
          v16 = (unsigned int)v38 + 1LL;
          if ( v16 > HIDWORD(v38) )
          {
            v27 = v11;
            v29 = v5;
            v32 = v9;
            sub_C8D5F0((__int64)&v37, v39, v16, 4u, v5, v9);
            v15 = (unsigned int)v38;
            v11 = v27;
            v5 = v29;
            v9 = v32;
          }
          ++v10;
          v37[v15] = v14;
          LODWORD(v38) = v38 + 1;
          if ( v9 == v10 )
            goto LABEL_15;
        }
        else
        {
          v12 = (unsigned int)v35;
          v13 = (unsigned int)v35 + 1LL;
          if ( v13 > HIDWORD(v35) )
          {
            v26 = v5;
            v28 = v9;
            v31 = v11;
            sub_C8D5F0((__int64)v11, v36, v13, 4u, v5, v9);
            v12 = (unsigned int)v35;
            v5 = v26;
            v9 = v28;
            v11 = v31;
          }
          ++v10;
          v34[v12] = v14;
          LODWORD(v35) = v35 + 1;
          if ( v9 == v10 )
          {
LABEL_15:
            v17 = v34;
            v18 = (unsigned int)v35;
            goto LABEL_16;
          }
        }
        v8 = *(_QWORD *)(v2 + 8);
      }
    }
    v17 = (unsigned int *)v36;
    v18 = 0;
LABEL_16:
    v30 = v5;
    sub_31DDD80(a1, v2, v17, v18, v5);
    sub_31DDD80(a1, v2, v37, (unsigned int)v38, v30);
    if ( v37 != (unsigned int *)v39 )
      _libc_free((unsigned __int64)v37);
    v19 = v34;
    if ( v34 != (unsigned int *)v36 )
      goto LABEL_19;
    return;
  }
  v20 = (__int64)(*(_QWORD *)(v2 + 16) - *(_QWORD *)(v2 + 8)) >> 5;
  v37 = (unsigned int *)v39;
  v38 = 0xC00000000LL;
  if ( (unsigned int)v20 > 0xCuLL )
  {
    v33 = v5;
    sub_C8D5F0((__int64)&v37, v39, (unsigned int)v20, 4u, v5, v6);
    LOBYTE(v5) = v33;
    v21 = &v37[(unsigned int)v38];
  }
  else
  {
    if ( !(_DWORD)v20 )
    {
      v25 = 0;
      v24 = 0;
      v23 = (unsigned int *)v39;
      goto LABEL_27;
    }
    v21 = (unsigned int *)v39;
  }
  v22 = 0;
  do
  {
    v21[v22] = v22;
    ++v22;
  }
  while ( (unsigned int)v20 != v22 );
  v23 = v37;
  v24 = v38 + v20;
  v25 = v24;
LABEL_27:
  LODWORD(v38) = v24;
  sub_31DDD80(a1, v2, v23, v25, v5);
  v19 = v37;
  if ( v37 != (unsigned int *)v39 )
LABEL_19:
    _libc_free((unsigned __int64)v19);
}
