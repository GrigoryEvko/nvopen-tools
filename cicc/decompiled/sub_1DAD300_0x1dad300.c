// Function: sub_1DAD300
// Address: 0x1dad300
//
void __fastcall sub_1DAD300(__int64 a1, __int64 a2, const __m128i *a3, __int64 a4, int a5, int a6)
{
  int v6; // r15d
  int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // r9d
  __int64 v12; // rdx
  unsigned int v13; // r15d
  __int64 v14; // rcx
  __int64 *v15; // rdi
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // [rsp+0h] [rbp-90h] BYREF
  _QWORD *v22; // [rsp+8h] [rbp-88h]
  __int64 v23; // [rsp+10h] [rbp-80h]
  _QWORD v24[15]; // [rsp+18h] [rbp-78h] BYREF

  v6 = (_DWORD)a4 << 31;
  v8 = sub_1DA81D0(a1, a3, (__int64)a3, a4, a5, a6);
  v12 = *(unsigned int *)(a1 + 296);
  v22 = v24;
  v21 = a1 + 216;
  v13 = v8 & 0x7FFFFFFF | v6;
  v23 = 0x400000000LL;
  if ( (_DWORD)v12 )
  {
    sub_1DAAC30((__int64)&v21, a2, v12, v9, v10, v11);
    v16 = (unsigned int)v23;
    if ( !(_DWORD)v23 )
      goto LABEL_8;
  }
  else
  {
    v14 = *(unsigned int *)(a1 + 300);
    if ( (_DWORD)v14 )
    {
      v15 = (__int64 *)(a1 + 224);
      v10 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3;
      do
      {
        if ( (*(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v15 >> 1) & 3) > (unsigned int)v10 )
          break;
        v12 = (unsigned int)(v12 + 1);
        v15 += 2;
      }
      while ( (_DWORD)v14 != (_DWORD)v12 );
    }
    v24[0] = a1 + 216;
    v16 = 1;
    LODWORD(v23) = 1;
    v24[1] = v14 | (v12 << 32);
  }
  if ( *((_DWORD *)v22 + 3) < *((_DWORD *)v22 + 2) )
  {
    v20 = (__int64)&v22[2 * v16 - 2];
    if ( *(_QWORD *)(*(_QWORD *)v20 + 16LL * *(unsigned int *)(v20 + 12)) == a2 )
    {
      sub_1DAB4F0((__int64)&v21, v13, v20, v14, v10);
      goto LABEL_11;
    }
  }
LABEL_8:
  v17 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v18 = (a2 >> 1) & 3;
  if ( v18 == 3 )
    v19 = *(_QWORD *)(v17 + 8) & 0xFFFFFFFFFFFFFFF9LL;
  else
    v19 = v17 | (2 * v18 + 2);
  sub_1DAD0A0((__int64)&v21, a2, v19, v13);
LABEL_11:
  if ( v22 != v24 )
    _libc_free((unsigned __int64)v22);
}
