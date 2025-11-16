// Function: sub_2B38EB0
// Address: 0x2b38eb0
//
void __fastcall sub_2B38EB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r14d
  unsigned int v6; // r15d
  int *v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  signed __int64 v10; // r13
  unsigned int v11; // ebx
  int v12; // eax
  __int64 v13; // rdi
  unsigned int v14; // esi
  _BYTE *v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 j; // rax
  _BYTE *v21; // rdi
  unsigned int v22; // esi
  __int64 v23; // rdi
  __int64 v24; // r9
  __int64 i; // r8
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  int *s2; // [rsp+8h] [rbp-B8h]
  int *s2a; // [rsp+8h] [rbp-B8h]
  __int64 v31[2]; // [rsp+10h] [rbp-B0h] BYREF
  _BYTE v32[48]; // [rsp+20h] [rbp-A0h] BYREF
  _BYTE *v33; // [rsp+50h] [rbp-70h] BYREF
  __int64 v34; // [rsp+58h] [rbp-68h]
  _BYTE v35[96]; // [rsp+60h] [rbp-60h] BYREF

  sub_2B32FB0(a2 + 112, a3, a3, a4);
  if ( *(_DWORD *)(a2 + 104) != 3 )
    return;
  v5 = *(_DWORD *)(a2 + 8);
  if ( !(unsigned __int8)sub_B4F6B0(*(_QWORD *)(a2 + 112), *(unsigned int *)(a2 + 120), v5) )
    return;
  v6 = *(_DWORD *)(a2 + 120);
  s2 = *(int **)(a2 + 112);
  if ( (unsigned __int8)sub_B4ED80(s2, v5, v5) )
    return;
  v10 = 4LL * v5;
  if ( v5 >= v6 )
  {
LABEL_10:
    v13 = *(_QWORD *)(a2 + 144);
    v14 = *(_DWORD *)(a2 + 152);
    v31[0] = (__int64)v32;
    v31[1] = 0xC00000000LL;
    sub_2B0FC00(v13, v14, (__int64)v31, (__int64)v7, v8, v9);
    sub_2B319A0((__int64)v31, *(int **)(a2 + 112), *(unsigned int *)(a2 + 120));
    v33 = v35;
    v17 = v10 >> 2;
    *(_DWORD *)(a2 + 152) = 0;
    v18 = v31[0];
    v34 = 0xC00000000LL;
    v19 = v10 >> 2;
    if ( (unsigned __int64)v10 > 0x30 )
    {
      sub_C8D5F0((__int64)&v33, v35, v10 >> 2, 4u, v16, v17);
      v17 = v10 >> 2;
      v15 = &v33[4 * (unsigned int)v34];
    }
    else
    {
      if ( !v10 )
      {
        v22 = 0;
        v21 = v35;
LABEL_16:
        LODWORD(v34) = v19;
        sub_2B0FC00((__int64)v21, v22, (__int64)v31, (__int64)v15, v16, v17);
        sub_2B38DA0((unsigned int *)a2, v31[0]);
        v23 = *(_QWORD *)(a2 + 112);
        v24 = v23 + 4LL * *(unsigned int *)(a2 + 120);
        for ( i = v23; v24 != v23; i += v10 )
        {
          v26 = v23;
          v23 += v10;
          if ( v23 != i )
          {
            v27 = 0;
            do
            {
              v28 = v27;
              *(_DWORD *)(v26 + 4 * v27) = v27;
              ++v27;
            }
            while ( (unsigned __int64)(v23 + -4 - i) >> 2 != v28 );
          }
        }
        if ( v33 != v35 )
          _libc_free((unsigned __int64)v33);
        if ( (_BYTE *)v31[0] != v32 )
          _libc_free(v31[0]);
        return;
      }
      v15 = v35;
    }
    for ( j = 0; j != v10; j += 4 )
      *(_DWORD *)&v15[j] = *(_DWORD *)(v18 + j);
    v21 = v33;
    LODWORD(v19) = v17 + v34;
    v22 = v17 + v34;
    goto LABEL_16;
  }
  v7 = s2;
  v11 = v5;
  while ( 1 )
  {
    if ( v10 )
    {
      s2a = v7;
      v12 = memcmp(&v7[v11], v7, v10);
      v7 = s2a;
      if ( v12 )
        break;
    }
    v11 += v5;
    if ( v6 <= v11 )
      goto LABEL_10;
  }
}
