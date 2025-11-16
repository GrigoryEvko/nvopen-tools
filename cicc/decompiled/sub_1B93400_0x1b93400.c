// Function: sub_1B93400
// Address: 0x1b93400
//
__int64 __fastcall sub_1B93400(unsigned __int64 a1, __int64 a2, int *a3)
{
  char v4; // al
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r15
  int v14; // eax
  __int64 v15; // rdx
  unsigned int v16; // r12d
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v20; // r13
  __int64 v21; // rax
  _BYTE *v22; // r9
  size_t v23; // rdx
  int v24; // edx
  int v25; // r10d
  unsigned __int64 v26; // [rsp+0h] [rbp-80h] BYREF
  __int128 v27; // [rsp+8h] [rbp-78h]
  __int128 v28; // [rsp+18h] [rbp-68h]
  __int64 v29; // [rsp+28h] [rbp-58h]
  _BYTE *v30; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+38h] [rbp-48h]
  _BYTE dest[64]; // [rsp+40h] [rbp-40h] BYREF

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 77 )
  {
    v5 = *(_QWORD *)(a1 + 24);
    v6 = *(unsigned int *)(v5 + 128);
    if ( (_DWORD)v6 )
    {
      v7 = *(_QWORD *)(v5 + 112);
      v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
      {
LABEL_4:
        if ( v9 != (__int64 *)(v7 + 16 * v6) )
        {
          v11 = 11LL * *((unsigned int *)v9 + 2);
          v12 = *(_QWORD *)(v5 + 136);
          v26 = 6;
          *(_QWORD *)&v27 = 0;
          v13 = v12 + 8 * v11;
          *((_QWORD *)&v27 + 1) = *(_QWORD *)(v13 + 24);
          if ( *((_QWORD *)&v27 + 1) != -8 && *((_QWORD *)&v27 + 1) != 0 && *((_QWORD *)&v27 + 1) != -16 )
            sub_1649AC0(&v26, *(_QWORD *)(v13 + 8) & 0xFFFFFFFFFFFFFFF8LL);
          v14 = *(_DWORD *)(v13 + 32);
          LODWORD(v28) = v14;
          *((_QWORD *)&v28 + 1) = *(_QWORD *)(v13 + 40);
          v15 = *(_QWORD *)(v13 + 48);
          v30 = dest;
          v29 = v15;
          v31 = 0x200000000LL;
          v16 = *(_DWORD *)(v13 + 64);
          if ( v16 && &v30 != (_BYTE **)(v13 + 56) )
          {
            v22 = dest;
            v23 = 8LL * v16;
            if ( v16 <= 2
              || (sub_16CD150((__int64)&v30, dest, v16, 8, v16, (int)dest),
                  v22 = v30,
                  (v23 = 8LL * *(unsigned int *)(v13 + 64)) != 0) )
            {
              memcpy(v22, *(const void **)(v13 + 56), v23);
            }
            LODWORD(v31) = v16;
            v14 = v28;
          }
          if ( (v14 & 0xFFFFFFFD) == 1 )
          {
            v17 = sub_22077B0(56);
            v18 = v17;
            if ( v17 )
            {
              *(_QWORD *)(v17 + 8) = 0;
              *(_QWORD *)(v17 + 16) = 0;
              *(_BYTE *)(v17 + 24) = 6;
              *(_QWORD *)(v17 + 32) = 0;
              *(_QWORD *)(v17 + 40) = a2;
              *(_QWORD *)(v17 + 48) = 0;
              *(_QWORD *)v17 = &unk_49F6EE8;
            }
          }
          else
          {
            v18 = 0;
          }
          if ( v30 != dest )
            _libc_free((unsigned __int64)v30);
          goto LABEL_15;
        }
      }
      else
      {
        v24 = 1;
        while ( v10 != -8 )
        {
          v25 = v24 + 1;
          v8 = (v6 - 1) & (v24 + v8);
          v9 = (__int64 *)(v7 + 16LL * v8);
          v10 = *v9;
          if ( a2 == *v9 )
            goto LABEL_4;
          v24 = v25;
        }
      }
    }
    v18 = 0;
    v29 = 0;
    v26 = 6;
    v30 = dest;
    v27 = 0;
    v28 = 0;
LABEL_15:
    sub_1455FA0((__int64)&v26);
    return v18;
  }
  v18 = 0;
  if ( v4 == 60 )
  {
    v26 = a1;
    *(_QWORD *)&v27 = a2;
    *(_QWORD *)&v28 = sub_1B8F9C0;
    *((_QWORD *)&v27 + 1) = sub_1B8E220;
    if ( (unsigned __int8)sub_1B932A0((__int64)&v26, a3, (__int64)a3) )
    {
      if ( *((_QWORD *)&v27 + 1) )
        (*((void (__fastcall **)(unsigned __int64 *, unsigned __int64 *, __int64))&v27 + 1))(&v26, &v26, 3);
      v20 = *(_QWORD *)sub_13CF970(a2);
      v21 = sub_22077B0(56);
      v18 = v21;
      if ( v21 )
      {
        *(_QWORD *)(v21 + 8) = 0;
        *(_QWORD *)(v21 + 16) = 0;
        *(_BYTE *)(v21 + 24) = 6;
        *(_QWORD *)(v21 + 32) = 0;
        *(_QWORD *)(v21 + 40) = v20;
        *(_QWORD *)(v21 + 48) = a2;
        *(_QWORD *)v21 = &unk_49F6EE8;
      }
    }
    else if ( *((_QWORD *)&v27 + 1) )
    {
      (*((void (__fastcall **)(unsigned __int64 *, unsigned __int64 *, __int64))&v27 + 1))(&v26, &v26, 3);
    }
  }
  return v18;
}
