// Function: sub_1AB25A0
// Address: 0x1ab25a0
//
unsigned __int64 __fastcall sub_1AB25A0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned __int64 v3; // r12
  _QWORD *v4; // rax
  __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // r14
  int v12; // eax
  __int64 v13; // r15
  unsigned __int64 v14; // rdx
  __int64 *v15; // r10
  __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rbx
  _QWORD *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v24; // r13
  __int64 v25; // r15
  _QWORD *v26; // rax
  __int64 *i; // rdx
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // r14
  __int64 *v33; // r13
  __int64 *v34; // r15
  __int64 v35; // rdi
  __int64 v36; // r8
  __int64 v38; // [rsp+10h] [rbp-F0h]
  __int64 v39; // [rsp+18h] [rbp-E8h]
  _QWORD *v40; // [rsp+18h] [rbp-E8h]
  char v41[16]; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v42; // [rsp+30h] [rbp-D0h]
  __int64 *v43; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+48h] [rbp-B8h]
  _WORD v45[88]; // [rsp+50h] [rbp-B0h] BYREF

  v3 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = (_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( ((a1 >> 2) & 1) == 0 )
    v4 = (_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) - 72);
  if ( *v4 )
  {
    v5 = v4[1];
    v6 = v4[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v6 = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
  }
  *v4 = a2;
  if ( a2 )
  {
    v7 = *(_QWORD *)(a2 + 8);
    v4[1] = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = (unsigned __int64)(v4 + 1) | *(_QWORD *)(v7 + 16) & 3LL;
    v4[2] = (a2 + 8) | v4[2] & 3LL;
    *(_QWORD *)(a2 + 8) = v4;
  }
  sub_1625C10(v3, 2, 0);
  sub_1625C10(v3, 23, 0);
  v8 = *(_QWORD *)(a2 + 24);
  if ( *(_QWORD *)(v3 + 64) != v8 )
  {
    v38 = *(_QWORD *)v3;
    v9 = *(__int64 **)(v8 + 16);
    v10 = *v9;
    *(_QWORD *)v3 = *v9;
    *(_QWORD *)(v3 + 64) = v8;
    v11 = *(_QWORD *)(a2 + 24);
    v12 = *(_DWORD *)(v11 + 12);
    if ( v12 != 1 )
    {
      v13 = 8;
      v39 = 8LL * (unsigned int)(v12 - 2) + 16;
      do
      {
        v19 = 3 * v13 - 24;
        v20 = *(_QWORD **)(v3 + v19 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
        v21 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + v13);
        if ( *v20 != v21 )
        {
          v45[0] = 257;
          v22 = sub_15FDBD0(47, (__int64)v20, v21, (__int64)&v43, v3);
          if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
            v14 = *(_QWORD *)(v3 - 8);
          else
            v14 = v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
          v15 = (__int64 *)(v14 + v19);
          if ( *(_QWORD *)(v14 + v19) )
          {
            v16 = v15[1];
            v17 = v15[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v17 = v16;
            if ( v16 )
              *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
          }
          *v15 = v22;
          if ( v22 )
          {
            v18 = *(_QWORD *)(v22 + 8);
            v15[1] = v18;
            if ( v18 )
              *(_QWORD *)(v18 + 16) = (unsigned __int64)(v15 + 1) | *(_QWORD *)(v18 + 16) & 3LL;
            v15[2] = (v22 + 8) | v15[2] & 3;
            *(_QWORD *)(v22 + 8) = v15;
          }
        }
        v13 += 8;
      }
      while ( v39 != v13 );
    }
    if ( *(_BYTE *)(v38 + 8) && v38 != v10 )
    {
      v24 = *(_QWORD *)(v3 + 8);
      v43 = (__int64 *)v45;
      v44 = 0x1000000000LL;
      if ( v24 )
      {
        v25 = 0;
        v26 = sub_1648700(v24);
        for ( i = (__int64 *)v45; ; i = v43 )
        {
          i[v25] = (__int64)v26;
          v25 = (unsigned int)(v44 + 1);
          LODWORD(v44) = v44 + 1;
          v24 = *(_QWORD *)(v24 + 8);
          if ( !v24 )
            break;
          v26 = sub_1648700(v24);
          if ( HIDWORD(v44) <= (unsigned int)v25 )
          {
            v40 = v26;
            sub_16CD150((__int64)&v43, v45, 0, 8, v28, v29);
            v25 = (unsigned int)v44;
            v26 = v40;
          }
        }
      }
      if ( *(_BYTE *)(v3 + 16) == 29 )
      {
        v30 = *(_QWORD *)(sub_1AA91E0(*(_QWORD **)(v3 + 40), *(_QWORD **)(v3 - 48), 0, 0) + 48);
        v31 = v30 - 24;
        if ( v30 )
        {
LABEL_36:
          v42 = 257;
          v32 = sub_15FDBD0(47, v3, v38, (__int64)v41, v31);
          if ( a3 )
            *a3 = v32;
          v33 = v43;
          v34 = &v43[(unsigned int)v44];
          if ( v43 != v34 )
          {
            do
            {
              v35 = *v33++;
              sub_1648780(v35, v3, v32);
            }
            while ( v34 != v33 );
            v34 = v43;
          }
          if ( v34 != (__int64 *)v45 )
            _libc_free((unsigned __int64)v34);
          return v3;
        }
      }
      else
      {
        v36 = *(_QWORD *)(v3 + 32);
        if ( v36 )
        {
          v31 = v36 - 24;
          goto LABEL_36;
        }
      }
      v31 = 0;
      goto LABEL_36;
    }
  }
  return v3;
}
