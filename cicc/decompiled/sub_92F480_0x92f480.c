// Function: sub_92F480
// Address: 0x92f480
//
__int64 __fastcall sub_92F480(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  __int64 *v6; // r15
  __int64 v7; // r13
  __m128i *v8; // r14
  __int64 v9; // rdx
  __m128i *v10; // rax
  __int64 v11; // rdi
  __m128i *v12; // rcx
  int v13; // eax
  __int64 v14; // rdi
  _BYTE *v15; // r13
  int v16; // eax
  char v18; // al
  unsigned int **v19; // rdi
  unsigned int v21; // [rsp+4h] [rbp-7Ch]
  __m128i *v23; // [rsp+8h] [rbp-78h]
  unsigned int v24; // [rsp+18h] [rbp-68h]
  _QWORD v25[4]; // [rsp+20h] [rbp-60h] BYREF
  char v26; // [rsp+40h] [rbp-40h]
  char v27; // [rsp+41h] [rbp-3Fh]

  v6 = *(__int64 **)(a2 + 72);
  v21 = a3;
  v7 = v6[2];
  v8 = sub_92CBF0(a1, (__int64)v6, a3);
  v10 = sub_92CBF0(a1, v7, v9);
  v11 = v8->m128i_i64[1];
  v12 = v10;
  v13 = *(unsigned __int8 *)(v11 + 8);
  if ( (unsigned int)(v13 - 17) <= 1 )
    LOBYTE(v13) = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
  if ( (unsigned __int8)v13 <= 3u || (_BYTE)v13 == 5 || (v13 & 0xFD) == 4 )
  {
    v14 = a1[1];
    v27 = 1;
    v25[0] = "cmp";
    v26 = 3;
    v15 = (_BYTE *)sub_B35C90(v14, a5, v8, v12, v25, 0, v24, 0);
    if ( unk_4D04700 && *v15 > 0x1Cu )
    {
      v16 = sub_B45210(v15);
      sub_B45150(v15, v16 | 1u);
    }
  }
  else
  {
    v23 = v12;
    v18 = sub_91B6F0(*v6);
    v19 = (unsigned int **)a1[1];
    v27 = 1;
    v25[0] = "cmp";
    v26 = 3;
    if ( v18 )
      v15 = (_BYTE *)sub_92B530(v19, a4, (__int64)v8, v23, (__int64)v25);
    else
      v15 = (_BYTE *)sub_92B530(v19, v21, (__int64)v8, v23, (__int64)v25);
  }
  return sub_92C930(a1, (__int64)v15, 0, *(_QWORD *)a2, (_DWORD *)(a2 + 36));
}
